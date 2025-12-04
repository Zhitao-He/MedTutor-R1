#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=SFT_Qwen2_medical
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --partition=airesearch
#SBATCH --account=airesearch
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

# --- Environment Setup ---
echo "Setting up the environment..."
source /project/airesearch/haolin/anaconda3/etc/profile.d/conda.sh
conda activate easyr1
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- FIX: Install DeepSpeed with the Correct CUDA Path ---
# This line tells the pip installer exactly where to find the CUDA toolkit
# that is compatible with your Conda environment. This resolves the build error.
# llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_medical.yaml
# bash examples/qwen2_5_vl_7b_AItutor_grpo_veto.sh
python ray_test.py
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Training Command ---
echo "Starting rule based AItutor training..."
