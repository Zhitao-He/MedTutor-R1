import json
import os

def process_teacher_io_file(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return None

        case_data = data[0]['input']['user_prompt']['static_context']['case_data']

        dialogue_history = []
        for round_data in data:
            students = {}
            for analysis in round_data['input']['user_prompt']['dynamic_context']['current_student_analyses']:
                students[analysis['student_id']] = analysis['analysis']

            patient_dialogue = ""
            for dialogue in reversed(round_data['input']['user_prompt']['dynamic_context']['dialogue_history']):
                if dialogue['speaker'] == 'Patient':
                    patient_dialogue = dialogue['content']
                    break
            
            teacher_guidance = round_data['output']['guidance']

            dialogue_history.append({
                "round": round_data['round'],
                "Patient": patient_dialogue,
                "students": students,
                "Teacher": teacher_guidance
            })

        file_id = os.path.splitext(os.path.basename(file_path))[0]

        return {
            "id": file_id,
            "case_data": case_data,
            "dialogue_history": dialogue_history
        }

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Error {file_path} {e}")
        return None

def process_all_files(root_dir):

    all_processed_data = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("Teacher_io.json"):
                file_path = os.path.join(dirpath, filename)
                processed_data = process_teacher_io_file(file_path)
                if processed_data:
                    all_processed_data.append(processed_data)

    return all_processed_data

base_path = 'Our_framework/Evaluation/Results/'
modal = 'Text'
your_target_directory = f'{base_path}/{modal}_test_Chunk_1_of_1'
all_data = process_all_files(your_target_directory)

output_filename = f'{base_path}/{modal}_processed_simulation_logs_all.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4) 

print(f"Done {output_filename}")