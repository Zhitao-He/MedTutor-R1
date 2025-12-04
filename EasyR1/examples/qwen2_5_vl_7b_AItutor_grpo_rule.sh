#!/bin/bash
set -x
RAY_TEMP_PATH="ray_tmp"
mkdir -p ${RAY_TEMP_PATH}
export RAY_TMPDIR=${RAY_TEMP_PATH}
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=
REWARD_FUNCTION_PATH=EasyR1/my_reward_functions/rule_based_reward.py
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=EasyR1/processed_datasets/medical_vqa_train.arrow \
    data.val_files=EasyR1/geometry3k/geometry3k-validation.arrow \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=4 \
    worker.reward.reward_type="sequential" \
    worker.reward.reward_function="${REWARD_FUNCTION_PATH}:calculate_comprehensive_reward" \
    trainer.find_last_checkpoint=false \
    worker.reward.reward_function_kwargs={} \
    data.filter_overlong_prompts_workers=4 \
    worker.rollout.limit_images=10 \
