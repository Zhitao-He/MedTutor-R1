import os
import json
from PIL import Image
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
import multiprocessing

# --- 1. 配置你的路径 ---

# 包含你的所有数据条目的单个 JSON 文件路径
INPUT_JSON_FILE = "/project/airesearch/haolin/medical/single_turn_formatted_data.json"

# 你的图片存放的根目录
BASE_IMAGE_PATH = "/project/airesearch/haolin/medical/MedXpertQA/images"

# 处理好的 .arrow 文件输出目录
OUTPUT_DIR = "/project/airesearch/haolin/EasyR1/processed_datasets"

# --- [优化] ---
# 设置用于并行处理的 CPU 核心数
NUM_PROCESSES = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() > 1 else 1

# --- 2. 脚本主逻辑 ---

def load_and_process_batch(batch):
    """
    这是一个用于 .map() 的函数，它接收一批数据并进行处理。
    这部分代码会在多个进程中并行运行。
    """
    loaded_images_batch = []
    user_prompts_batch = []
    assistant_answers_batch = []

    for images_paths, messages in zip(batch['images'], batch['messages']):
        # --- a. 加载图片 ---
        loaded_images = []
        if images_paths:
            for rel_path in images_paths:
                image_filename = os.path.basename(rel_path)
                full_image_path = os.path.join(BASE_IMAGE_PATH, image_filename)
                if os.path.exists(full_image_path):
                    try:
                        img = Image.open(full_image_path).convert("RGB")
                        loaded_images.append(img)
                    except Exception:
                        pass
        
        # --- b. 提取对话并应用过滤逻辑 ---
        if images_paths and not loaded_images:
            continue

        if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
            
            # --- c. 正确构建 user_prompt ---
            original_user_content = messages[0]['content']
            cleaned_user_content = original_user_content.replace("<image>\n", "").replace("<image>", "")
            num_loaded_images = len(loaded_images)
            if num_loaded_images > 0:
                image_tokens = " ".join(["<image>"] * num_loaded_images) + "\n"
                final_user_prompt = image_tokens + cleaned_user_content
            else:
                final_user_prompt = cleaned_user_content
            
            assistant_answer = messages[1]['content']

            loaded_images_batch.append(loaded_images)
            user_prompts_batch.append(final_user_prompt)
            assistant_answers_batch.append(assistant_answer)

    return {
        "images": loaded_images_batch,
        "problem": user_prompts_batch,
        "answer": assistant_answers_batch
    }

def create_dataset():
    """
    使用并行处理从JSON文件高效创建Hugging Face Dataset。
    """
    print(f"开始加载 JSON 文件: {INPUT_JSON_FILE}")
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"错误：指定的 JSON 文件不存在: {INPUT_JSON_FILE}")
        return
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            all_samples_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 JSON 文件: {e}")
        return
    if not isinstance(all_samples_data, list):
        print("错误: JSON 文件的顶层结构不是一个列表。")
        return
    
    print(f"在文件中找到了 {len(all_samples_data)} 个样本。")
    print(f"将使用 {NUM_PROCESSES} 个CPU核心进行并行处理。")

    initial_dataset = Dataset.from_list(all_samples_data)

    final_features = Features({
        'images': Sequence(HFImage()),
        'problem': Value('string'),
        'answer': Value('string')
    })

    print("\n开始并行处理数据（加载图片并格式化prompt）...")
    processed_dataset = initial_dataset.map(
        load_and_process_batch,
        batched=True,
        batch_size=100,
        num_proc=NUM_PROCESSES,
        remove_columns=initial_dataset.column_names,
        features=final_features
    )

    print(f"\n成功处理了 {len(processed_dataset)} 个样本。")
    print("\n数据集结构:")
    print(processed_dataset)
    
    # --- 步骤 4. 保存为 Arrow 文件 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_output_file = os.path.join(OUTPUT_DIR, "medical_vqa_train.arrow")

    print(f"\n训练集样本总数: {len(processed_dataset)}")
    print(f"正在保存完整的训练集到: {train_output_file}")
    
    # --- [核心修复] ---
    # 步骤 1: 将 Hugging Face Dataset 转换为 Pandas DataFrame
    print("正在将数据集转换为 Pandas DataFrame...")
    df = processed_dataset.to_pandas()
    
    # 步骤 2: 使用 Pandas DataFrame 的 .to_feather() 方法保存为 Arrow 文件
    print(f"正在使用 Pandas 将 DataFrame 保存为 Feather (Arrow) 文件: {train_output_file}")
    df.to_feather(train_output_file)

    print("\n处理完成！")

if __name__ == "__main__":
    create_dataset()


