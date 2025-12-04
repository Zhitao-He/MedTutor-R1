#!/bin/bash
set -x

# Set environment variables
export TMPDIR="/tmp/${USER}/tmp"
export RAY_TEMP_DIR="/tmp/${USER}/ray_tmp"
mkdir -p "${TMPDIR}"
mkdir -p "${RAY_TEMP_DIR}"
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=

# Safety measure: Clean up any old Ray processes before starting
echo "✅ Ensuring any old Ray processes are stopped..."
ray stop --force

# Define model path
MODEL_PATH=SFT/output/qwen2_5vl-7b/sft_Med_AITutor

# Run the python script, which will now manage the Ray cluster itself
echo "✅ Starting training..."
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=EasyR1/processed_datasets/medical_vqa_train.arrow \
    data.val_files= \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_AITutor_SFT+RL \
    trainer.n_gpus_per_node=4 \
    worker.reward.reward_type="batch" \
    worker.reward.reward_function="EasyR1/my_reward_functions/ai_tutor_reward.py:compute_scores_in_batch" \
    trainer.find_last_checkpoint=false \
    worker.reward.reward_function_kwargs='{"max_workers": 16}' \
    data.filter_overlong_prompts_workers=4 \
    worker.rollout.limit_images=10

echo "✅ Script finished."
