export PYTHONUNBUFFERED=1
export N_GPUS=1
export ROLLOUT_TP_SIZE=1
export BASE_MODEL=adamame/model/sft_distill1.5b
export DATA_DIR=adamame/2_rl/data
export EXPERIMENT_NAME=grpo_distill1.5b
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_ATTN_IMPLEMENTATION=sdpa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LR=1e-5
export LORA_RANK=32
export LORA_ALPHA=64
export KL_COEF=0.001
export VAL_SAMPLES_TO_PRINT=${VAL_SAMPLES_TO_PRINT:-2}
export VAL_SAMPLE_MAX_CHARS=${VAL_SAMPLE_MAX_CHARS:-240}
export HYDRA_FULL_ERROR=1

# Set up environment
module load anaconda3/2024.2
conda activate rl_env
export MKL_THREADING_LAYER=GNU
export LANG_LOG_FILE=adamame/logs/${EXPERIMENT_NAME}/lang_dist.jsonl
export WANDB_API_KEY=

# Set up data
train_path=$DATA_DIR/train/train_dapomath17k_5k.parquet
val_path=$DATA_DIR/val/val_dapomath17k.parquet
train_files="['$train_path']"
test_files="['$val_path']"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.enable=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    +actor_rollout_ref.actor.use_dr_grpo=True \
    +actor_rollout_ref.actor.dr_grpo_max_tokens=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='rl_grpo' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.use_tqdm=True \
    +trainer.val_samples_to_print=$VAL_SAMPLES_TO_PRINT \
    +trainer.val_sample_max_chars=$VAL_SAMPLE_MAX_CHARS \
    +trainer.val_before_train=True \
    +trainer.train_generations_to_log_to_wandb=8 \
    +trainer.log_format_reward=False \
    +trainer.use_format_reward=False \
    trainer.resume_mode=auto \
    trainer.total_epochs=10 $@
