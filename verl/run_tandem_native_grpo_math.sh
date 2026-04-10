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

export WANDB_ENTITY=difanjiao
export WANDB_PROJECT=tandem-native-grpo-math
export WANDB_API_KEY=f510b3737ade928e3e94556e9fae86fcbd716dc2

set -x
SCRATCH_DIR=/datadrive/difan/verl-llm-tandem/scratch

B=8
VAL_B=512
N=8
L=3000
VAL_L=3000
SENIOR_MODEL=${SENIOR_MODEL:-Qwen/Qwen3-4B}
JUNIOR_MODEL=${JUNIOR_MODEL:-${SENIOR_MODEL}}

# "bernoulli" (token-level) or "sentence" (sentence-level round-robin).
# NOTE: sentence strategy is NOT recommended for MATH — the period token (id=13)
# appears inside decimals (e.g. 3.14) and triggers spurious boundaries mid-number.
# Use bernoulli for token-level tandem on MATH. If you want sentence-level,
# set boundary_token_ids explicitly to newline-only via engine_kwargs.
TANDEM_STRATEGY=${TANDEM_STRATEGY:-bernoulli}

SENIOR_TAG=$(echo ${SENIOR_MODEL} | sed 's|.*/||')
JUNIOR_TAG=$(echo ${JUNIOR_MODEL} | sed 's|.*/||')
if [ "$SENIOR_MODEL" = "$JUNIOR_MODEL" ]; then
    MODEL_TAG=${SENIOR_TAG}
else
    MODEL_TAG=${SENIOR_TAG}_jr-${JUNIOR_TAG}
fi

if [ "$TANDEM_STRATEGY" = "sentence" ]; then
    NAME=tandem_sentence_grpo_math_${MODEL_TAG}
else
    NAME=tandem_native_grpo_math_${MODEL_TAG}
fi

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=$SCRATCH_DIR/MATH/hendrycks_math/train.parquet \
    data.val_files=$SCRATCH_DIR/MATH/math500/test.parquet \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_response_length=$VAL_L \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.enabled=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.frozen_model=${JUNIOR_MODEL} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.prob_primary=0.5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.selection_strategy=${TANDEM_STRATEGY} \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.tandem_config.frozen_gpu_devices=[1]' \
    +actor_rollout_ref.actor.tandem_jr_tkn_weight=0.15 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    +reward_model.custom_reward_function.path=verl/utils/reward_score/math_dataset.py \
    +reward_model.custom_reward_function.name=compute_score \
    trainer.logger='[console,wandb]' \
    trainer.project_name=tandem-native-grpo-math \
    trainer.experiment_name=${NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SCRATCH_DIR/checkpoints/${NAME} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    +ray_init.num_cpus=16 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.resume_mode='auto' \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +trainer.start_save_step=20 \
    +trainer.best_hf_checkpoint_dir=$SCRATCH_DIR/hf/${NAME}/best_model \
    $@
