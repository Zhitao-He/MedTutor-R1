import json
import os
from tqdm import tqdm


simulation_log_path = "Our_framework/Simulation_log"

template_file_path = 'Our_framework/SFT_files/instruct_teacher_template.txt'
with open(template_file_path, 'r', encoding='utf-8') as f:
    template = f.read()

total_files = 0
for dir_name in os.listdir(simulation_log_path):
    if dir_name.startswith(("MM", "Text")):
        path_level1 = os.path.join(simulation_log_path, dir_name)
        if not os.path.isdir(path_level1):
            continue
        for chunk_dir_name in os.listdir(path_level1):
            if "test_Chunk" in chunk_dir_name:
                path_level2 = os.path.join(path_level1, chunk_dir_name)
                if not os.path.isdir(path_level2):
                    continue
                for session_dir_name in os.listdir(path_level2):
                    path_level3 = os.path.join(path_level2, session_dir_name)
                    if not os.path.isdir(path_level3):
                        continue
                    for filename in os.listdir(path_level3):
                        if filename.endswith("Teacher_io.json"):
                            total_files += 1


all_results = []
statistics = {
    "total_dialogues": 0,
    "by_path_level1": {}
}

pbar = tqdm(total=total_files, desc="", unit="file")

for dir_name in os.listdir(simulation_log_path):
    if dir_name.startswith(("MM", "Text")):
        path_level1 = os.path.join(simulation_log_path, dir_name)
        if not os.path.isdir(path_level1):
            continue
        
        statistics["by_path_level1"][dir_name] = {
            "total_dialogues": 0,
            "files_processed": 0,
            "errors": 0
        }
        
        for chunk_dir_name in os.listdir(path_level1):
            if "test_Chunk" in chunk_dir_name:
                path_level2 = os.path.join(path_level1, chunk_dir_name)
                if not os.path.isdir(path_level2):
                    continue
                for session_dir_name in os.listdir(path_level2):
                    path_level3 = os.path.join(path_level2, session_dir_name)
                    if not os.path.isdir(path_level3):
                        continue
                    for filename in os.listdir(path_level3):
                        if filename.endswith("Teacher_io.json"):
                            teacher_file_path = os.path.join(path_level3, filename)
                            
                            pbar.set_description(f"processing: {os.path.basename(teacher_file_path)}")
                            
                            try:
                                with open(teacher_file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                rounds = {}
                                for item in data:
                                    round_num = item['round']
                                    if round_num not in rounds:
                                        rounds[round_num] = []
                                    rounds[round_num].append(item)
                                
                                file_dialogues_count = 0
                                for round_num, round_data in rounds.items():
                                    first_item = round_data[0]
                                    last_item = round_data[-1]
                                    
                                    input_data = first_item['input']['user_prompt']
                                    case_data = input_data['static_context']['case_data']
                                    dialogue_history = input_data['dynamic_context']['dialogue_history']
                                    student_analyses = input_data['dynamic_context']['current_student_analyses']
                                    
                                    current_analysis_contents = [analysis['analysis'] for analysis in student_analyses]
                                    
                                    round_dialogues = {}
                                    current_round_count = 1
                                    current_round_dialogues = []
                                    student_ids = {analysis['student_id'] for analysis in student_analyses}

                                    
                                    for entry in dialogue_history:
                                        if (entry['visibility'] in ['student&patient', 'teacher&student'] and 
                                            not entry['content'].startswith('(query_for_patient)') and
                                            entry['speaker'] not in ['Expert (Review)', 'Supervisor (Review)']):
                                            
                                            if entry['speaker'] in student_ids and entry['content'] in current_analysis_contents:
                                                continue
                                            
                                            current_round_dialogues.append(entry)
                                            
                                            if entry['speaker'] in ['Teacher', 'Teacher (Draft 1)', 'Teacher (Draft 2)']:
                                                round_dialogues[f"Round {current_round_count}"] = current_round_dialogues.copy()
                                                current_round_count += 1
                                                current_round_dialogues = []
                                    
                                    if current_round_dialogues:
                                        round_dialogues[f"Round {current_round_count}"] = current_round_dialogues
                                    
                                    formatted_history = ""
                                    for round_name, dialogues in round_dialogues.items():
                                        formatted_history += f"\n{round_name}:\n"
                                        for entry in dialogues:
                                            formatted_history += f"- {entry['speaker']}: {entry['content']}\n"
                                    
                                    formatted_analyses = ""
                                    for analysis in student_analyses:
                                        formatted_analyses += f"- {analysis['student_id']}: {analysis['analysis']}\n"
                                    
                                    filled_template = template.replace('{{case_question}}', case_data['case_question'])
                                    filled_template = filled_template.replace('{{case_question_answer}}', case_data['case_question_answer'])
                                    filled_template = filled_template.replace('<image>', ', '.join(case_data['case_images']))
                                    filled_template = filled_template.replace('{{dialogue_history}}', formatted_history)
                                    filled_template = filled_template.replace('{{student_analyses}}', formatted_analyses)
                                    
                                    internal_monologue = first_item['output']['internal_monologue']
                                    
                                    if 'revised_guidance' in last_item['output']:
                                        revised_guidance = last_item['output']['revised_guidance']
                                    else:
                                        revised_guidance = last_item['output']['guidance']
                                    
                                    assistant_content = f"{internal_monologue}\n\n{revised_guidance}"
                                    
                                    image_paths = [f"mllm_demo_data/{image}" for image in case_data['case_images']]
                                    
                                    case_images = case_data.get('case_images', [])

                                    if case_images:
                                        user_content = f"<image>{filled_template}"
                                        image_paths = [f"mllm_demo_data/{image}" for image in case_images]
                                    else:
                                        user_content = filled_template
                                        image_paths = []

                                    round_result = {
                                        "messages": [
                                            {
                                                "content": user_content,
                                                "role": "user"
                                            },
                                            {
                                                "content": assistant_content,
                                                "role": "assistant"
                                            }
                                        ],
                                        "images": image_paths,
                                        "source_file": teacher_file_path,
                                        "path_level1": dir_name
                                    }
                                    
                                    all_results.append(round_result)
                                    file_dialogues_count += 1
                                    statistics["total_dialogues"] += 1
                                    statistics["by_path_level1"][dir_name]["total_dialogues"] += 1
                                
                                statistics["by_path_level1"][dir_name]["files_processed"] += 1
                                pbar.set_postfix({
                                    'dialogues': file_dialogues_count,
                                    'total': statistics["total_dialogues"]
                                })
                                
                            except Exception as e:
                                print(f"\nError {teacher_file_path} {str(e)}")
                                statistics["by_path_level1"][dir_name]["errors"] += 1
                            finally:
                                pbar.update(1)

pbar.close()


output_file_path = 'Our_framework/SFT_files/single_formatted_data.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)


stats_file_path = 'Our_framework/SFT_files/processing_statistics.json'
with open(stats_file_path, 'w', encoding='utf-8') as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)
