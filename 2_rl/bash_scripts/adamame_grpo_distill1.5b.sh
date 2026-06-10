#!/bin/sh

#SBATCH --job-name=adamame_grpo_distill1.5b
#SBATCH --output=adamame/logs/adamame_grpo_distill1.5b.log
#SBATCH --time=1-00:00:00
#SBATCH --gres=

export PYTHONUNBUFFERED=1

VERL_DIR=adamame/2_rl
EXPERIMENT_NAME=adamame_grpo_distill1.5b
CHECKPOINT_DIR="$VERL_DIR/checkpoints/$EXPERIMENT_NAME"

ALIGN_BONUS=2.0 bash "$VERL_DIR/stage2_scripts/trainer/${EXPERIMENT_NAME}.sh" && \
STEP=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt") && \
python "$VERL_DIR/scripts/merge_lora_checkpoint.py" \
    --actor_dir "$CHECKPOINT_DIR/global_step_${STEP}/actor" \
    --base_model DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_dir "$VERL_DIR/model/$EXPERIMENT_NAME";
rm -rf "$CHECKPOINT_DIR";
