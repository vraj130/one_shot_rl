#!/bin/bash
set -x

export CHECKPOINTS_DIR="./checkpoints"

# Use single GPU training to avoid NCCL peer access issues
export CUDA_VISIBLE_DEVICES=0

# Add PyTorch memory optimization settings
# More aggressive CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.2,max_split_size_mb:32

# Disable PyTorch inductor to avoid compilation errors
export TORCH_COMPILE_BACKEND=eager
export TORCH_INDUCTOR_DISABLE=1

# Set VLLM backend
export VLLM_ATTENTION_BACKEND=XFORMERS

# Ray settings
export RAY_DEDUP_LOGS=0
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_ADDRESS="local"

MODEL_PATH="/root/one_shot_rl/pretrained_checkpoints/Qwen-2-8b/checkpoints/Qwen2.5-1.5B"

SAVE_DIR="./training_checkpoints/"

# Activate the .oneshot virtual environment
source ~/.oneshot/bin/activate

# Apply 8-bit quantization patch
python -c "from verl.utils.model_loading_patch import patch_vllm_model_loading; patch_vllm_model_loading()"

# Set bitsandbytes environment variables for 8-bit quantization
export BNB_CUDA_VERSION=120
export BNB_USE_RUNTIME_KERNEL=1

# Set distributed training debug info
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_BACKEND=gloo

# Run the training script
python -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/one_shot_rlvr/arc_agi_2_train.parquet \
 data.val_files=data/test/arc_agi_2_eval.parquet \
 data.train_batch_size=1 \
 data.val_batch_size=1 \
 data.max_prompt_length=512 \
 data.max_response_length=640 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path="$MODEL_PATH" \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=hf \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
 actor_rollout_ref.rollout.n=1 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console'] \
 trainer.project_name='arc_1_shot'\
 trainer.experiment_name='Qwen2_one_shot_rl'\
 trainer.checkpoints_dir="$SAVE_DIR" \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=21 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=25 2>&1 | tee verl_demo.log