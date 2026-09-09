#!/usr/bin/env bash
# ============================================================================
# KL-Reg GRPO on DeepScaleR: the regularisation ablation.
#   Solo rollouts, as in train/vanilla_grpo.sh, plus a per-token KL penalty in
#   the loss toward the frozen pre-RL base. verl loads the reference policy from
#   actor_rollout_ref.model.path, which is $BASE_MODEL, so the reference is the
#   same frozen model the tandem arm uses as its junior. The arm exists to
#   separate the two explanations of the tandem effect: staying near the base
#   distribution, against being co-generated with a partner.
#
#   Differences from train/vanilla_grpo.sh, and this is the whole list:
#     - actor.use_kl_loss=True. This alone spawns the reference worker, which is
#       what verl's need_reference_policy() keys on.
#     - actor.kl_loss_coef=$KL_COEF, the swept beta. Required, no default: the
#       arm is meaningless without a stated coefficient.
#     - actor.kl_loss_type=low_var_kl, the k3 estimator, verl's GRPO standard.
#     - ref.fsdp_config.param_offload=True, so the reference weights sit in host
#       memory between forwards. Costs RAM, costs no GPU.
#   algorithm.use_kl_in_reward stays False: this is a loss-term regulariser, not
#   reward shaping. The step cost grows by one reference forward per step.
#
#   Launch:  KL_COEF=0.001 bash train/kl_reg.sh
#   Smoke:   KL_COEF=0.001 TOTAL_STEPS=3 SAVE_FREQ=2 TEST_FREQ=3 \
#            EXP_NAME=klreg_smoke bash train/kl_reg.sh
# ============================================================================
set -xeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
[ -f "$HERE/env.sh" ] || { echo "[error] $HERE/env.sh not found: copy train/env.sh.example to train/env.sh and edit it" >&2; exit 1; }
# shellcheck disable=SC1091
source "$HERE/env.sh"

# ----------------------------- KNOBS ---------------------------------------
POLICY_MODEL=${POLICY_MODEL:-$BASE_MODEL}
GPUS=${GPUS:-0,1}
TRAIN_GPUS=${TRAIN_GPUS:-2}

KL_COEF=${KL_COEF:?"set KL_COEF explicitly (the swept beta, e.g. 0.001)"}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}     # kl | abs | mse | low_var_kl | full
KL_TAG=$(echo "$KL_COEF" | tr '.' 'p')       # 0.001 -> 0p001, for the experiment name

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
ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.8}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
OFFLOAD=${OFFLOAD:-True}
EXP_NAME=${EXP_NAME:-kl_reg_qwen3_4b_deepscaler_b${KL_TAG}}
CKPT_DIR=${CKPT_DIR:-$CKPT_ROOT/${EXP_NAME}}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}
# ---------------------------------------------------------------------------

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true   # some schedulers export it; verl rejects it alongside CUDA_VISIBLE_DEVICES
# Under a scheduler, keep the allocation it assigned rather than forcing 0,1.
if [ -z "${SLURM_JOB_ID:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPUS; fi

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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
