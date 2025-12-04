import json
import base64
import os
import time
import sys
import concurrent.futures

from openai import OpenAI


API_KEY = os.getenv('OPENAI_API_KEY', '') 
IMAGE_DIRECTORY = 'dataset/MedXpertQA/images'


def load_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error {filepath}")
        return None

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        if 'eval_result' in filepath:
            return []
        print(f"Error {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error {filepath}")
        return None

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error {image_path}")
        return None

def call_judge_model(client, judge_prompt, item_data):
    if not client:
        print("Error API")
        return None
    item_data_str = json.dumps(item_data, indent=2)
    user_message_content = [{"type": "text", "text": item_data_str}]
    image_filenames = item_data.get("case_data", {}).get("case_images", [])
    for filename in image_filenames:
        image_path = os.path.join(IMAGE_DIRECTORY, filename)
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            user_message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_message_content}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4.1", messages=messages, max_tokens=1024, response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def call_judge_with_retry(client, judge_prompt, item_data, max_retries=3):
    for attempt in range(max_retries):
        response = call_judge_model(client, judge_prompt, item_data)
        if response is not None:
            return response
        print(f"{item_data.get('id', 'N/A')} try {attempt + 1}/{max_retries} fails")
        time.sleep(1)
    return None

def evaluate_item(item, client, judge_prompts):
    print(f"--- Eval: {item['id']} ---")
    item_scores = {"id": item['id']}
    item_failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_judge = {
            executor.submit(call_judge_with_retry, client, prompt, item): judge_type
            for judge_type, prompt in judge_prompts.items()
        }
        for future in concurrent.futures.as_completed(future_to_judge):
            judge_type = future_to_judge[future]
            try:
                response_content = future.result()
                if response_content:
                    score_data = json.loads(response_content)
                    score = score_data.get(f"{judge_type}_Score")
                    justification = score_data.get(f"{judge_type}_Justification")
                    item_scores[judge_type] = {"Score": score, "Justification": justification}
                else:
                    item_failed = True; break
            except Exception:
                item_failed = True; break
    if item_failed: return item['id']
    else: return item_scores

def calculate_and_print_stats(scores_list, judge_types):
    if not scores_list: print("\nNone"); return
    totals = {key: 0 for key in judge_types}; counts = {key: 0 for key in judge_types}
    for item_scores in scores_list:
        for judge_type in judge_types:
            score = item_scores.get(judge_type, {}).get("Score")
            if isinstance(score, (int, float)):
                totals[judge_type] += score; counts[judge_type] += 1
    print(f"Total: {len(scores_list)}")
    for judge_type in judge_types:
        if counts[judge_type] > 0:
            average = totals[judge_type] / counts[judge_type]
            print(f"  - {judge_type} Mean: {average:.2f} (Total {counts[judge_type]})")
        else:
            print(f"  - {judge_type} Mean: N/A ")
    print("--------------------")

def process_evaluation_file(input_filepath, output_filepath, client, judge_prompts):
    evaluation_data = load_json_file(input_filepath)
    if not evaluation_data:
        print(f"Fails: {input_filepath}")
        return

    all_results = load_json_file(output_filepath)
    completed_ids = {item['id'] for item in all_results if isinstance(item, dict)}
    
    if all_results:
        print(f"Loaded {len(completed_ids)}")

    items_to_process = [item for item in evaluation_data if item['id'] not in completed_ids]
    
    if not items_to_process:
        calculate_and_print_stats(all_results, judge_prompts.keys())
        return
    
    print(f"Total: {len(evaluation_data)}, Done: {len(completed_ids)}, Processing: {len(items_to_process)}")

    failed_item_ids = []
    MAX_CONCURRENT_ITEMS = 5
    progress_count = 0
    original_order_map = {item['id']: i for i, item in enumerate(evaluation_data)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ITEMS) as executor:
        future_to_item = {
            executor.submit(evaluate_item, item, client, judge_prompts): item
            for item in items_to_process
        }

        for future in concurrent.futures.as_completed(future_to_item):
            result = future.result()
            progress_count += 1
            print(f"进度: ({progress_count}/{len(items_to_process)})")
            
            if isinstance(result, dict):
                all_results.append(result)
                
                all_results.sort(key=lambda x: original_order_map.get(x['id'], float('inf')))
                
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=4)
                except Exception as e:
                    print(f" Error {result['id']}  {e}")
            else:
                failed_item_ids.append(result)


    print(f"\n[INFO] Done: {os.path.basename(input_filepath)}")
    if all_results:
        calculate_and_print_stats(all_results, judge_prompts.keys())

    if failed_item_ids:
        print(f"\n[FAILED]  {len(failed_item_ids)}:")
        print("- " + "\n- ".join(failed_item_ids))


def main():
    judge_prompts = {
        "ETS": load_text_file('Our_framework/Evaluation/Judgement/ETS_judge.txt'),
        "MPS": load_text_file('Our_framework/Evaluation/Judgement/MPS_judge.txt'),
        "MSM": load_text_file('Our_framework/Evaluation/Judgement/MSM_judge.txt')
    }
    if not all(judge_prompts.values()):
        print("Failed")
        return

    client = None
    if API_KEY and API_KEY != 'sk-xxx':
        client = OpenAI(base_url='', api_key=API_KEY)
    else:
        print("Error API")
        return

    root_directory = 'Our_framework/Evaluation/Results/Main_result/'

    try:
        subdirectories = sorted([entry for entry in os.scandir(root_directory) if entry.is_dir()], key=lambda e: e.name)
        for entry in subdirectories:
            sub_dir_path = entry.path
            sub_dir_name = entry.name
            input_file_path = None
            for file_entry in os.scandir(sub_dir_path):
                if file_entry.is_file() and file_entry.name.endswith('logs_all.json'):
                    input_file_path = file_entry.path
                    break
            
            if input_file_path:
                output_file_path = os.path.join(sub_dir_path, f"{sub_dir_name}_eval_result.json")
                
                process_evaluation_file(input_file_path, output_file_path, client, judge_prompts)
            else:
                print(f"\nError '{entry.name}'")
    except FileNotFoundError:
        print(f"Error: {root_directory}")

if __name__ == '__main__':
    main()