#!/usr/bin/env bash
# ============================================================================
# TANDEM GRPO on DeepScaleR: the treatment arm.
#   Senior   $BASE_MODEL, trained.
#   Junior   $BASE_MODEL again, frozen. The junior is a copy of the weights the
#            senior starts from, so the pair is self-paired at step 0.
#   Rollout  handoff at word boundaries: a fair coin redraws the active model
#            after any token in train/assets/qwen3_word_boundary_ids.json, and
#            MAX_GAP_TOKENS forces a redraw if no boundary token has appeared.
#   Reward   reward/math_boxed_reward.py over the whole co-generated response,
#            so the junior's tokens shape the reward the senior is trained on.
#   Loss     GRPO over the senior's tokens only (JR_TKN_WEIGHT=0). The vLLM
#            fork attaches an authorship mask to every completion; verl's
#            ppo_loss gates response_mask by it in workers/utils/losses.py and
#            logs the realised split as actor/tandem_senior_token_frac.
#   Val      senior alone. The agent loop sets tandem_disabled on validation, so
#            the reported val pass@1 is a solo number, not a team number.
#   Layout   two GPUs, senior and junior on separate cards, never shared:
#            cuda:0 trains and hosts the rollout engine, cuda:1 holds the frozen
#            junior and its KV cache. verl pins the rollout server to the worker
#            group's cards, so the second card is reachable only by listing it
#            in VLLM_TANDEM_ALL_GPUS, and FROZEN_GPU indexes into that widened
#            list. This is why TRAIN_GPUS is 1 while the job holds 2 GPUs.
#   Ckpt     $CKPT_ROOT/$EXP_NAME, HF weights under hf/global_step_N.
#
#   Hyperparameters are identical to train/vanilla_grpo.sh. Its header lists
#   every deliberate difference between the two arms.
#
#   Launch:  bash train/tandem_grpo.sh          (2 GPUs)
#   Smoke:   TOTAL_STEPS=3 SAVE_FREQ=2 TEST_FREQ=3 EXP_NAME=tandem_smoke \
#            bash train/tandem_grpo.sh
#            then check actor/tandem_senior_token_frac lands in [0.3, 0.7]. If
#            the metric never appears, the mask never arrived: the rollout is
#            running under stock vLLM and this is plain GRPO.
# ============================================================================
set -xeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
[ -f "$HERE/env.sh" ] || { echo "[error] $HERE/env.sh not found: copy train/env.sh.example to train/env.sh and edit it" >&2; exit 1; }
# shellcheck disable=SC1091
source "$HERE/env.sh"

# ----------------------------- KNOBS ---------------------------------------
POLICY_MODEL=${POLICY_MODEL:-$BASE_MODEL}
JUNIOR_MODEL=${JUNIOR_MODEL:-$BASE_MODEL}
GPUS=${GPUS:-0,1}
TRAIN_GPUS=${TRAIN_GPUS:-1}

# --- TANDEM block: the only structural difference from the GRPO arm ---
PROB_PRIMARY=${PROB_PRIMARY:-0.5}
SELECTION=${SELECTION:-word}
MAX_GAP_TOKENS=${MAX_GAP_TOKENS:-32}
BOUNDARY_IDS_PATH=${BOUNDARY_IDS_PATH:-$REPO/train/assets/qwen3_word_boundary_ids.json}
FROZEN_GPU=${FROZEN_GPU:-1}
JR_TKN_WEIGHT=${JR_TKN_WEIGHT:-0}            # lambda_jun. 0 trains on the senior's tokens only; >0 gives the junior's tokens that weight in the gradient

LR=${LR:-1e-6}
CLIP_RATIO=${CLIP_RATIO:-0.2}
TRAIN_BATCH=${TRAIN_BATCH:-16}
MINI_BATCH=${MINI_BATCH:-8}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_RESPONSE=${MAX_RESPONSE:-3000}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-0.6}
PPO_MAX_TOKEN=${PPO_MAX_TOKEN:-10000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-null}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMP=${VAL_TEMP:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_N=${VAL_N:-1}
ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.65}   # not 0.8: cuda:0 carries the engine and the single-rank FSDP
                                             # peaks (about 18 GB) at once. On a 95 GB card 0.8 gave
                                             # 76.9 GB of engine plus 17.7 GB of training and OOM'd.
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
OFFLOAD=${OFFLOAD:-True}
EXP_NAME=${EXP_NAME:-tandem_grpo_qwen3_4b_deepscaler}
CKPT_DIR=${CKPT_DIR:-$CKPT_ROOT/${EXP_NAME}}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}
# ---------------------------------------------------------------------------

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true   # some schedulers export it; verl rejects it alongside CUDA_VISIBLE_DEVICES
# Under a scheduler, keep the allocation it assigned rather than forcing 0,1.
if [ -z "${SLURM_JOB_ID:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPUS; fi
# The verl fork widens the rollout server's visible devices with any card listed
# here that the worker group does not own, which is how the junior reaches the
# second GPU. Without it, frozen_gpu_devices=[1] is an invalid device ordinal.
export VLLM_TANDEM_ALL_GPUS="${CUDA_VISIBLE_DEVICES:-$GPUS}"
echo "VLLM_TANDEM_ALL_GPUS=$VLLM_TANDEM_ALL_GPUS"

# ==== TANDEM TRIGGER ========================================================
# This environment variable is the entire interface. The vLLM fork reads it in
# EngineArgs.create_tandem_config; there is no CLI flag and no hydra key. Set,
# the frozen junior co-generates every training rollout and the authorship mask
# rides the batch into the loss gate. Unset, this script is
# train/vanilla_grpo.sh with one card training.
TANDEM_JSON="{\"enabled\": true, \"frozen_model\": \"$JUNIOR_MODEL\", \"selection_strategy\": \"$SELECTION\", \"prob_primary\": $PROB_PRIMARY, \"max_gap_tokens\": $MAX_GAP_TOKENS, \"frozen_gpu_devices\": [$FROZEN_GPU], \"boundary_token_ids_path\": \"$BOUNDARY_IDS_PATH\"}"
export VLLM_TANDEM_CONFIG="$TANDEM_JSON"
echo "VLLM_TANDEM_CONFIG=$VLLM_TANDEM_CONFIG"
# ===========================================================================

train_files=${TRAIN_FILES:-$DATA_ROOT/deepscaler/train.parquet}
# Validation is a held-out slice of the training distribution, never a reported
# benchmark: a checkpoint selected on a benchmark is selected on its own test set.
# data/build_deepscaler.py carves it and removes those rows from train.parquet.
val_files="[$DATA_ROOT/deepscaler/heldout.parquet]"
reward_fn=$REPO/reward/math_boxed_reward.py
max_num_tokens=$(( MAX_PROMPT + MAX_RESPONSE + 1 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${TRAIN_BATCH} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESPONSE} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$POLICY_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.tandem_jr_tkn_weight=${JR_TKN_WEIGHT} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMP} \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    model_engine=dp \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$reward_fn" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='[console,wandb]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${TRAIN_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP} \
    +trainer.persist_hf_model=True \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "$@"
