#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate tandem-verl

export HF_HOME=/datadrive/difan/verl-llm-tandem/scratch/models

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

SECRETS_FILE="$(dirname "$(readlink -f "$0")")/../scratch/wandb_secrets.env"
if [ -f "$SECRETS_FILE" ]; then
    set -a; source "$SECRETS_FILE"; set +a
else
    echo "[error] wandb secrets not found at $SECRETS_FILE" >&2; exit 1
fi
export WANDB_PROJECT=$WANDB_PROJECT_TANDEM_NATIVE_GRPO_DEEPSCALER

set -x
SCRATCH_DIR=/datadrive/difan/verl-llm-tandem/scratch

# [thinking-suppression] Senior (Qwen3-4B-Instruct-2507)'s default chat_template
# has no `enable_thinking` conditional, so passing enable_thinking=False would
# be silently dropped. Junior (Qwen3-0.6B, hybrid) emits <think> 5/5 trials at
# temp=0.6 when its first chunk starts at a fresh assistant turn.
# Fix: copy junior's chat_template (which contains the conditional) into the
# senior tokenizer's tokenizer_config.json. Verified that the two templates
# produce byte-identical prompts on single-turn data when enable_thinking is
# not passed; only with enable_thinking=False does the junior template add
# the '<think>\n\n</think>\n\n' prefill that suppresses junior's leak.
# Idempotent. Reversible by re-downloading the senior from HF.
python -c "
import json, glob
sr = glob.glob('${SCRATCH_DIR}/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer_config.json')[0]
jr = glob.glob('${SCRATCH_DIR}/models/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer_config.json')[0]
d = json.load(open(sr))
new_tpl = json.load(open(jr))['chat_template']
if d['chat_template'] != new_tpl:
    d['chat_template'] = new_tpl
    json.dump(d, open(sr, 'w'), indent=2, ensure_ascii=False)
    print('[setup] patched senior chat_template ->', sr)
else:
    print('[setup] senior chat_template already patched')
"


B=16
VAL_B=512
N=8
L=3000
VAL_L=3000
SENIOR_MODEL=${SENIOR_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
# JUNIOR_MODEL=${JUNIOR_MODEL:-${SENIOR_MODEL}}
JUNIOR_MODEL=Qwen/Qwen3-0.6B

# "bernoulli" (token-level), "sentence" (\n\n-round-robin), or
# "word" (Qwen3 BPE Ġ-prefix round-robin + max_gap_tokens fallback).
# sentence uses \n\n-ending tokens (primarily '.\n\n'=382, ~17 switches/resp).
# word uses Ġ-prefix tokens (~hundreds of switches/resp on prose); fallback
# forces a switch after TANDEM_MAX_GAP consecutive non-boundary tokens so that
# LaTeX/math atomic blocks can't be monopolized by one model.
# boundary_token_ids is auto-resolved in vllm_rollout_spmd.py when not set.
TANDEM_STRATEGY=${TANDEM_STRATEGY:-word}
TANDEM_MAX_GAP=${TANDEM_MAX_GAP:-32}

SENIOR_TAG=$(echo ${SENIOR_MODEL} | sed 's|.*/||')
JUNIOR_TAG=$(echo ${JUNIOR_MODEL} | sed 's|.*/||')
if [ "$SENIOR_MODEL" = "$JUNIOR_MODEL" ]; then
    MODEL_TAG=${SENIOR_TAG}
else
    MODEL_TAG=${SENIOR_TAG}_jr-${JUNIOR_TAG}
fi

if [ "$TANDEM_STRATEGY" = "sentence" ]; then
    NAME=tandem_sentence_grpo_deepscaler_at4_${MODEL_TAG}_no-jr-tkn-weight
elif [ "$TANDEM_STRATEGY" = "word" ]; then
    NAME=tandem_word_grpo_deepscaler_at4_${MODEL_TAG}_gap${TANDEM_MAX_GAP}_no-jr-tkn-weight
else
    NAME=tandem_native_grpo_deepscaler_at4_${MODEL_TAG}_no-jr-tkn-weight
fi

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=$SCRATCH_DIR/MATH/deepscaler/train.parquet \
    "data.val_files=[$SCRATCH_DIR/MATH/amc_23_25/test.parquet,$SCRATCH_DIR/MATH/aime_24_26/test.parquet,$SCRATCH_DIR/MATH/minerva/test.parquet]" \
    '+data.apply_chat_template_kwargs={enable_thinking: false}' \
    data.train_batch_size=$B \
    data.val_batch_size=$VAL_B \
    data.max_prompt_length=1024 \
    data.max_response_length=$L \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${SENIOR_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_response_length=$VAL_L \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.enabled=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.frozen_model=${JUNIOR_MODEL} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.prob_primary=0.5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.selection_strategy=${TANDEM_STRATEGY} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.max_gap_tokens=${TANDEM_MAX_GAP} \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.frozen_gpu_devices=[1]' \
    +actor_rollout_ref.actor.tandem_jr_tkn_weight=0.00 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    +reward_model.custom_reward_function.path=verl/utils/reward_score/math_dataset.py \
    +reward_model.custom_reward_function.name=compute_score \
    trainer.logger='[console,wandb]' \
    trainer.project_name=tandem-native-grpo-deepscaler \
    trainer.experiment_name=${NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SCRATCH_DIR/checkpoints/${NAME} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=2 \
    +ray_init.num_cpus=16 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.resume_mode='auto' \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +trainer.start_save_step=20 \
    +trainer.best_hf_checkpoint_dir=$SCRATCH_DIR/hf/${NAME}/best_model \
    '+trainer.val_macro_avg_sources=[amc_23_25,aime_24_26,minerva]' \
    +trainer.best_hf_metric_key=val-core/macro_avg/acc/pass@4 \
    $@
