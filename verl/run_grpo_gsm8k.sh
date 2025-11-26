#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl

export HF_HOME=/datadrive/difan/verl-llm-tandem/scratch/models
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export WANDB_ENTITY=difanjiao
export WANDB_PROJECT=vanilla-grpo-gsm8k
export WANDB_API_KEY=f510b3737ade928e3e94556e9fae86fcbd716dc2

set -x
SCRATCH_DIR=/datadrive/difan/verl-llm-tandem/scratch

B=16
VAL_B=512
N=8
L=1024
NAME=vanilla_grpo_gsm8k_Qwen3-0.6B

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_files=$SCRATCH_DIR/data/gsm8k/train.parquet \
    data.val_files=$SCRATCH_DIR/data/gsm8k/test.parquet \
    data.train_batch_size=$B \
    data.val_batch_size=$VAL_B \
    data.max_prompt_length=512 \
    data.max_response_length=$L \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=40000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=vanilla-grpo-gsm8k \
    trainer.experiment_name=${NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SCRATCH_DIR/checkpoints/${NAME} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    +ray_init.num_cpus=16 \
    trainer.val_before_train=True \
    trainer.log_val_generations=$VAL_B \
    trainer.resume_mode='disable' \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +trainer.start_save_step=10 \
    $@
