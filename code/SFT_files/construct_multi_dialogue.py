import json
import os
import random
from tqdm import tqdm

simulation_log_path = "Our_framework/Simulation_log"
template_file_path = 'Our_framework/SFT_files/instruct_teacher_template_multi.txt'
output_file_path = 'Our_framework/SFT_files/multi_turn_augmented_data.json'
stats_file_path = 'Our_framework/SFT_files/multi_turn_augmented_statistics.json'

try:
    with open(template_file_path, 'r', encoding='utf-8') as f:
        template = f.read()
except FileNotFoundError:
    print(f"Error {template_file_path}")
    exit()

target_files = []
if os.path.exists(simulation_log_path):
    for dir_name in os.listdir(simulation_log_path):
        if dir_name.startswith(("MM", "Text")):
            path_level1 = os.path.join(simulation_log_path, dir_name)
            if not os.path.isdir(path_level1): continue
            for chunk_dir_name in os.listdir(path_level1):
                if "test_Chunk" in chunk_dir_name:
                    path_level2 = os.path.join(path_level1, chunk_dir_name)
                    if not os.path.isdir(path_level2): continue
                    for session_dir_name in os.listdir(path_level2):
                        path_level3 = os.path.join(path_level2, session_dir_name)
                        if not os.path.isdir(path_level3): continue
                        for filename in os.listdir(path_level3):
                            if filename.endswith("Teacher_io.json"):
                                target_files.append((os.path.join(path_level3, filename), dir_name))
else:
    print(f"Error {simulation_log_path}")
    exit()

print(f"{len(target_files)} target files")

all_results = []
statistics = {
    "total_files_processed": 0,
    "total_sessions_generated": 0,
    "total_turns_generated": 0,
    "session_round_distribution": {},
    "by_path_level1": {}
}

pbar = tqdm(total=len(target_files), desc="processing", unit="file")
for teacher_file_path, dir_name in target_files:
    pbar.set_description(f"Processing: {os.path.basename(teacher_file_path)}")
    
    statistics["by_path_level1"].setdefault(dir_name, {"files_processed": 0, "sessions_generated": 0, "errors": 0})
    
    try:
        with open(teacher_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rounds = {}
        for item in data:
            round_num = item['round']
            rounds.setdefault(round_num, []).append(item)
        
        multi_turn_messages = []
        
        for round_num in sorted(rounds.keys()):
            round_data = rounds[round_num]
            first_item = round_data[0]
            last_item = round_data[-1]
            
            input_data = first_item['input']['user_prompt']
            case_data = input_data['static_context']['case_data']
            dialogue_history = input_data['dynamic_context']['dialogue_history']
            student_analyses = input_data['dynamic_context']['current_student_analyses']
            case_images_list = case_data.get('case_images', [])

            formatted_analyses = "\n".join([f"- {analysis['student_id']}: {analysis['analysis']}" for analysis in student_analyses])

            user_content = ""
            if round_num == 1:
                filled_template = template.replace('{{case_question}}', case_data.get('case_question', 'N/A'))
                filled_template = filled_template.replace('{{case_question_answer}}', case_data.get('case_question_answer', 'N/A'))
                
                initial_patient_complaint = ""
                for entry in dialogue_history:
                    if entry['speaker'] == 'Patient' and entry['content'] != 'No question from students':
                        initial_patient_complaint = f"\n## Current Dialogue\n- {entry['speaker']}: {entry['content']}\n"
                        break
                        
                initial_student_analyses_section = f"\n## Current Student Analyses\n{formatted_analyses}"
                
                user_content = filled_template + initial_patient_complaint + initial_student_analyses_section
                
                if case_images_list: 
                    user_content = f"<image>{user_content}"
            else:
                new_dialogue_section = ""
                last_teacher_index = max((i for i, entry in enumerate(dialogue_history) if entry['speaker'].startswith('Teacher')), default=-1)
                new_dialogue_entries = dialogue_history[last_teacher_index + 1:]

                full_interaction = []
                for entry in new_dialogue_entries:
                    is_student_query = '(query_for_patient)' in entry['content']
                    is_valid_patient_statement = (entry['speaker'] == 'Patient' and entry['content'] != 'No question from students')

                    if is_student_query or is_valid_patient_statement:
                         full_interaction.append(f"- {entry['speaker']}: {entry['content']}")


                if full_interaction:
                    new_dialogue_section = "\n".join(full_interaction)
                
                user_content = (
                    "## Current Dialogue\n" + 
                    (new_dialogue_section if new_dialogue_section else "No new patient-student dialogue.") + 
                    "\n\n## Current Student Analyses\n" + 
                    formatted_analyses
                )

            multi_turn_messages.append({"role": "user", "content": user_content.strip()})
            
            internal_monologue = first_item['output']['internal_monologue']
            revised_guidance = last_item['output'].get('revised_guidance')
            if not revised_guidance: revised_guidance = last_item['output'].get('guidance', 'No guidance provided.')
            assistant_content = f"{internal_monologue}\n\n{revised_guidance}"
            multi_turn_messages.append({"role": "assistant", "content": assistant_content})
        num_rounds = len(rounds)
        image_paths = [f"mllm_demo_data/{image}" for image in case_images_list] if case_images_list else []
        
        def create_and_log_session(messages, round_count, suffix=""):
            source_file_info = f"{teacher_file_path} ({suffix})" if suffix else teacher_file_path
            session = {"messages": messages, "images": image_paths, "source_file": source_file_info, "path_level1": dir_name}
            all_results.append(session)
            
            key = f"{round_count}_rounds"
            statistics["session_round_distribution"][key] = statistics["session_round_distribution"].get(key, 0) + 1
            statistics["total_sessions_generated"] += 1
            statistics["total_turns_generated"] += round_count
            statistics["by_path_level1"][dir_name]["sessions_generated"] += 1
        
        if num_rounds >= 4:
            if random.random() < 0.20:
                create_and_log_session(multi_turn_messages[:4], 2, "rounds 1-2")
                create_and_log_session(multi_turn_messages[:6], 3, "rounds 1-3")
                create_and_log_session(multi_turn_messages, num_rounds, "all rounds")
            else:
                create_and_log_session(multi_turn_messages[:4], 2, "rounds 1-2")
                create_and_log_session(multi_turn_messages, num_rounds, "all rounds")
        elif num_rounds == 3:
            create_and_log_session(multi_turn_messages[:4], 2, "rounds 1-2")
            create_and_log_session(multi_turn_messages, num_rounds, "all rounds")
        else:
            create_and_log_session(multi_turn_messages, num_rounds)

        statistics["total_files_processed"] += 1
        statistics["by_path_level1"][dir_name]["files_processed"] += 1

    except Exception as e:
        print(f"\nError {teacher_file_path}  {str(e)}")
        statistics["by_path_level1"][dir_name]["errors"] = statistics["by_path_level1"][dir_name].get("errors", 0) + 1
    finally:
        pbar.update(1)
pbar.close()


with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

with open(stats_file_path, 'w', encoding='utf-8') as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)

