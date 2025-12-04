# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import json
# import os 

# import ray
# from omegaconf import OmegaConf

# from ..single_controller.ray import RayWorkerGroup
# from ..utils.tokenizer import get_processor, get_tokenizer
# from ..workers.fsdp_workers import FSDPWorker
# from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
# from .config import PPOConfig
# from .data_loader import create_dataloader
# from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# # please make sure main_task is not scheduled on head
# @ray.remote(num_cpus=1)
# class Runner:
#     """A runner for RL training."""

#     def run(self, config: PPOConfig):
#         # print config
#         print(json.dumps(config.to_dict(), indent=2))

#         # instantiate tokenizer
#         tokenizer = get_tokenizer(
#             config.worker.actor.model.model_path,
#             override_chat_template=config.data.override_chat_template,
#             trust_remote_code=config.worker.actor.model.trust_remote_code,
#             use_fast=True,
#         )
#         processor = get_processor(
#             config.worker.actor.model.model_path,
#             override_chat_template=config.data.override_chat_template,
#             trust_remote_code=config.worker.actor.model.trust_remote_code,
#             use_fast=True,
#         )

#         # define worker classes
#         ray_worker_group_cls = RayWorkerGroup
#         role_worker_mapping = {
#             Role.ActorRolloutRef: ray.remote(FSDPWorker),
#             Role.Critic: ray.remote(FSDPWorker),
#         }
#         global_pool_id = "global_pool"
#         resource_pool_spec = {
#             global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
#         }
#         mapping = {
#             Role.ActorRolloutRef: global_pool_id,
#             Role.Critic: global_pool_id,
#         }
#         resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

#         if config.worker.reward.reward_type == "sequential":
#             RewardManager = SequentialFunctionRewardManager
#         elif config.worker.reward.reward_type == "batch":
#             RewardManager = BatchFunctionRewardManager
#         else:
#             raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

#         RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
#         reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
#         val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

#         train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

#         trainer = RayPPOTrainer(
#             config=config,
#             tokenizer=tokenizer,
#             processor=processor,
#             train_dataloader=train_dataloader,
#             val_dataloader=val_dataloader,
#             role_worker_mapping=role_worker_mapping,
#             resource_pool_manager=resource_pool_manager,
#             ray_worker_group_cls=ray_worker_group_cls,
#             reward_fn=reward_fn,
#             val_reward_fn=val_reward_fn,
#         )
#         trainer.init_workers()
#         trainer.fit()


# def main():
#     cli_args = OmegaConf.from_cli()
#     default_config = OmegaConf.structured(PPOConfig())

#     if hasattr(cli_args, "config"):
#         config_path = cli_args.pop("config", None)
#         file_config = OmegaConf.load(config_path)
#         default_config = OmegaConf.merge(default_config, file_config)

#     ppo_config = OmegaConf.merge(default_config, cli_args)
#     ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
#     ppo_config.deep_post_init()

#     if not ray.is_initialized():
#         runtime_env = {
#             "env_vars": {
#                 "TOKENIZERS_PARALLELISM": "true",
#                 "NCCL_DEBUG": "WARN",
#                 "VLLM_LOGGING_LEVEL": "WARN",
#                 "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
#                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
#                 "PYTHONUNBUFFERED": "1",
#                 "CUDA_DEVICE_MAX_CONNECTIONS": "1",
#             }
#         }
#         plasma_dir = "/project/airesearch/hyangby/ray_plasma" # 换成你自己的路径！

#         # 2. 用代码确保这个目录存在
#         os.makedirs(plasma_dir, exist_ok=True)
#         node_ip_address = os.environ.get("RAY_IP")
#         ray.init(
#             _node_ip_address=node_ip_address,
#             runtime_env=runtime_env,
#             _plasma_directory=plasma_dir
#         )

#     runner = Runner.remote()
#     ray.get(runner.run.remote(ppo_config))

#     if ppo_config.trainer.ray_timeline is not None:
#         # use `export RAY_PROFILING=1` to record the ray timeline
#         ray.timeline(filename=ppo_config.trainer.ray_timeline)


# if __name__ == "__main__":
#     main()

#手动启用ray
import json
import os
import socket # --- ADD THIS IMPORT ---
import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# Runner class remains exactly the same...
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""
    # ... (no changes inside this class) ...
    def run(self, config: PPOConfig):
        # ...
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    # ... (config loading remains the same) ...
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    # --- START OF FINAL MODIFICATION ---
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        }
        plasma_dir = f"/project/airesearch/{os.environ.get('USER', 'default_user')}/ray_plasma"
        os.makedirs(plasma_dir, exist_ok=True)
        
        # Explicitly find the machine's IP address to prevent hangs
        host_name = socket.gethostname()
        node_ip_address = socket.gethostbyname(host_name)
        print(f"✅ Starting Ray... Detected Node IP: {node_ip_address}")
        
        # Let this script start and manage the Ray cluster
        ray.init(
            _node_ip_address=node_ip_address, # Use the detected IP
            runtime_env=runtime_env,
            _plasma_directory=plasma_dir
            # We REMOVED address="auto" because this script is now the master
        )
    # --- END OF FINAL MODIFICATION ---

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    # The ray.stop() will be handled automatically when the script exits
    if ppo_config.trainer.ray_timeline is not None:
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()