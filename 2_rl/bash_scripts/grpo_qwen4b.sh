#!/bin/sh

#SBATCH --job-name=grpo_qwen4b
#SBATCH --output=adamame/logs/grpo_qwen4b.log
#SBATCH --time=1-00:00:00
#SBATCH --gres=

export PYTHONUNBUFFERED=1

VERL_DIR=adamame/2_rl
EXPERIMENT_NAME=grpo_qwen4b
CHECKPOINT_DIR="$VERL_DIR/checkpoints/$EXPERIMENT_NAME"

ALIGN_BONUS=2.0 bash "$VERL_DIR/stage2_scripts/trainer/${EXPERIMENT_NAME}.sh" && \
STEP=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt") && \
python "$VERL_DIR/scripts/merge_lora_checkpoint.py" \
    --actor_dir "$CHECKPOINT_DIR/global_step_${STEP}/actor" \
    --base_model Qwen/Qwen3-4B \
    --output_dir "$VERL_DIR/model/$EXPERIMENT_NAME";
rm -rf "$CHECKPOINT_DIR";
