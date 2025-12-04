import json
import base64
import os
import time
import re
from openai import OpenAI
from pathlib import Path


client = OpenAI(
    base_url='',
    api_key=os.getenv("OPENAI_API_KEY", '')
)


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error at {image_path}")
        return None

def load_prompt_template(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_json_from_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return response_text


def get_llm_response(prompt_template, data_point, image_paths):

    try:

        label = data_point.get("label")
        options = data_point.get("options")
        correct_answer_text = "Answer not available"
        if label and options and label in options:
            correct_answer_text = options[label]
        
        original_qa_data = {
            "question": data_point.get("question", ""),
            "answer": correct_answer_text,
            "images": data_point.get("images", [])
        }
        
        socratic_steps_data = data_point.get("model_response", [])

        prompt_data = {
            "original_qa": original_qa_data,
            "socratic_steps": socratic_steps_data
        }

        data_point_str = json.dumps(prompt_data, indent=2, ensure_ascii=False) 
        full_prompt = prompt_template.replace("{data_point_json}", data_point_str)
        content_parts = [{"type": "text", "text": full_prompt}]
        
        for image_path in image_paths:
            base64_image = image_to_base64(image_path)
            if base64_image:
                image_type = Path(image_path).suffix.lower().replace('.', '')
                if image_type == 'jpg': image_type = 'jpeg'
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}
                })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=4096,
        )
        
        raw_response_text = response.choices[0].message.content
        parsed_response = extract_json_from_response(raw_response_text)
        
        return parsed_response
    
    except Exception as e:
        print(f"Error {data_point.get('id', 'N/A')}: {str(e)}")
        return None

def main():
    prompt_file = "Our_framework/Patient_simulate/patient_script.txt"
    json_file = "Our_framework/question_decomposition/processed_data/MedXpert_processed_results_MM.json"
    base_image_dir = "dataset/MedXpertQA/images"
    output_file = "Our_framework/Patient_simulate/MedXpert_patient_script_MM.json"

    prompt_template = load_prompt_template(prompt_file)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f) 
        print(f"Total {len(all_data)} records")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Loading '{json_file}' Error: {e}")
        return

    results = []
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_ids = {item['id'] for item in results}
            print(f"Loaded {len(results)} ")
        except (json.JSONDecodeError, TypeError):
            print(f"Error {output_file}")
            results = []
            
    for i, item in enumerate(all_data, 1):
        item_id = item.get("id")
        if item_id in processed_ids:
            continue

        print(f"\n=====[{i}/{len(all_data)}] Processing ID: {item_id}=====")

        image_paths = []
        if "images" in item and item["images"]:
            for img_filename in item["images"]:
                image_paths.append(os.path.join(base_image_dir, img_filename))
            print(f"Find {len(image_paths)} images: {item['images']}")
        else:
            print("Error")

        response_content = get_llm_response(prompt_template, item, image_paths)
        
        if response_content:
            new_result = item.copy()
            
            if isinstance(response_content, (dict, list)):
                new_result["patient_script"] = response_content 
                print(f"ID {item_id} Done")
            else:
                print(f"Error {item_id}")
                new_result["model_response"] = {"error": "Failed to parse model output as JSON."}
                new_result["model_raw_output"] = response_content

            results.append(new_result)
            
            temp_output_file = output_file + '.tmp'
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            os.replace(temp_output_file, output_file)
        else:
            print(f"ID {item_id} Error")
        
        time.sleep(1)
    
    print(f"\nDone {output_file}")


if __name__ == "__main__":
    main()