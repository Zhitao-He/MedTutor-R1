#!/bin/bash

# --- SLURM Directives ---
# A descriptive name for your job to easily identify it
#SBATCH --job-name=SFT_Qwen3_8B_medical
#
# Request computational resources
#SBATCH --nodes=1                     # Request one node
#SBATCH --cpus-per-task=48            # Number of CPU cores per task
#SBATCH --mem=384G                    # Memory request for the node
#
# Specify job queue and account details
#SBATCH --partition=airesearch        # The partition (queue) to run on
#SBATCH --account=airesearch          # Your account name for resource allocation
#
# Set a time limit for the job
#SBATCH --time=08:00:00               # Set a time limit of 8 hours (HH:MM:SS)
#
# Define log files for output and errors
#SBATCH --output=slurm-%j.out         # Standard output log file (%j is the job ID)

echo "Setting up the environment..."
source /project/airesearch/haolin/anaconda3/etc/profile.d/conda.sh
conda activate easyr1
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# 设置实验的主目录
EXPERIMENT_DIR="/project/airesearch/haolin/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_AITutor_SFT+Vanilla_RL"

# 定义需要合并的所有step
STEPS="15 30"

# --- 开始循环合并 ---

echo "🚀 Starting batch model merge process..."
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Target Steps: $STEPS"
echo "=================================================="

# 循环遍历每一个step
for STEP in $STEPS; do
  # --- 动态生成当前step的路径 ---
  
  # FSDP检查点（checkpoint）的完整路径
  FSDP_CHECKPOINT_DIR="$EXPERIMENT_DIR/global_step_${STEP}/actor"
  
  # 合并后模型的输出路径 (注意：脚本会将模型保存在 FSDP_CHECKPOINT_DIR/huggingface/ 中)
  OUTPUT_INFO_DIR="$FSDP_CHECKPOINT_DIR/huggingface"

  echo "Processing merge for step ${STEP}..."
  echo "  FSDP Checkpoint (Input): $FSDP_CHECKPOINT_DIR"
  echo "  Consolidated Model (Output) will be saved in: $OUTPUT_INFO_DIR"

  # --- 运行合并脚本 (已修正) ---
  
  # 只传递脚本认识的 --local_dir 参数
  python model_merger.py \
      --local_dir "$FSDP_CHECKPOINT_DIR"
  
  # 检查上一个命令是否成功执行
  if [ $? -eq 0 ]; then
    echo "✅ Merge complete for step ${STEP}. Consolidated model saved to: $OUTPUT_INFO_DIR"
  else
    echo "❌ Error during merge for step ${STEP}. Please check the logs."
  fi
  
  echo "--------------------------------------------------"
done

echo "🎉 All merge tasks have been completed."