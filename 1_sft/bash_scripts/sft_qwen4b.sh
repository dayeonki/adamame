#!/bin/sh

#SBATCH --job-name=sft_qwen4b
#SBATCH --output=../log/sft_qwen4b.out
#SBATCH --error=../log/sft_qwen4b.error
#SBATCH --time=2-00:00:00
#SBATCH --mem=32gb
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gres=


module load cuda/13.1.1
export CUDA_HOME=/opt/common/cuda/cuda-13.1.1
export PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VIRTUAL_ENV/lib:$VIRTUAL_ENV/lib64:$LD_LIBRARY_PATH"

export HF_HOME="/.cache"
export HF_DATASETS="/.cache"
export TRANSFORMERS_CACHE="/.cache"
export HF_TOKEN=""
export FORCE_TORCHRUN=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_TORCH_COMPILE=0

llamafactory-cli train stage1_scripts/sft_qwen4b.yaml && \
llamafactory-cli export stage1_scripts/merge_qwen4b.yaml && \
