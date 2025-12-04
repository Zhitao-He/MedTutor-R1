import os
from datasets import load_dataset

# --- 配置 ---

# 1. 你想要下载的数据集列表
dataset_names = [
    "hiyouga/math12k",
    "hiyouga/geometry3k",
    "hiyouga/journeybench-multi-image-vqa",
    "hiyouga/rl-mixed-dataset"
]

# 2. 你希望将数据集缓存到的服务器路径
#    datasets 库会把数据文件下载到这个目录下的 "huggingface/datasets" 子目录中
target_cache_dir = "/project/airesearch/haolin/EasyR1"

# --- 下载脚本 ---

print(f"开始下载数据集，将缓存到: {target_cache_dir}")

# 确保目标目录存在
os.makedirs(target_cache_dir, exist_ok=True)

for name in dataset_names:
    print("\n" + "="*50)
    print(f"正在处理数据集: {name}")
    try:
        # 使用 cache_dir 参数指定下载位置
        load_dataset(name, cache_dir=target_cache_dir)
        print(f"✅ 成功下载或验证: {name}")
    except Exception as e:
        print(f"❌ 下载失败: {name}")
        print(f"错误信息: {e}")

print("\n" + "="*50)
print("所有数据集处理完毕！")