import os
import json
import random
from openai import OpenAI
from copy import deepcopy
import datetime
import time     
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

class AI_Client:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("API key is required.")
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, system_prompt: str, user_prompt_data: dict, max_retries: int = 3) -> dict:

        for attempt in range(max_retries):
            try:
                user_content = []
                image_data_list = user_prompt_data.pop("images_data", []) 
                user_prompt_string = json.dumps(user_prompt_data, indent=2, ensure_ascii=False)
                user_content.append({"type": "text", "text": user_prompt_string})

                for img_data in image_data_list:
                    user_content.append({
                        "type": "image_url",
                        "image_url": { "url": f"data:image/jpeg;base64,{img_data}" }
                    })

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"}
                )
                response_content = completion.choices[0].message.content
                parsed_json = json.loads(response_content)
                return parsed_json
            except Exception as e:
                error_message = f"An API call to model '{self.model_name}' failed: {e}"
                print(f"[RETRYABLE ERROR] Attempt {attempt + 1}/{max_retries}: {error_message}")
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 2
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        print(f"[FATAL ERROR] API call failed after {max_retries} attempts.")
        return {"error": f"API call failed after {max_retries} retries."}

class Orchestrator:
    def __init__(self, case_data: dict, persona: dict, student_profiles: list, ai_config: dict, prompts_dir: str, save_dir):
        self.case_data = case_data
        self.persona = persona
        self.students = student_profiles
        
        self.dialogue_history = {}
        self.current_round_number = 0
        self.current_patient_statement = self.case_data.get("patient_script", {}).get("patient_fact_base", {}).get("chief_complaint", "Doctor, I need help.")
        self.agent_io_logs = {}

        case_id = self.case_data.get('id', 'UnknownCase')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(save_dir, f"{case_id}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] Log files for this run will be saved in: {self.output_dir}")

        self.prompt_templates = self._load_prompts(prompts_dir)
        print("All prompt templates loaded successfully.")


        api_key = ai_config['api_key']
        base_url = ai_config['base_url']
        models = ai_config['models']
        
        self.student_ai = AI_Client(models['student'], api_key, base_url)
        self.teacher_ai = AI_Client(models['teacher'], api_key, base_url)
        self.patient_ai = AI_Client(models['patient'], api_key, base_url)
        self.expert_ai = AI_Client(models['expert'], api_key, base_url)
        self.supervisor_ai = AI_Client(models['supervisor'], api_key, base_url)
        
        print("Orchestrator initialized with all AI clients.")

    def log_agent_io(self, agent_id: str, system_prompt: str, user_prompt: dict, output: dict):

        if agent_id not in self.agent_io_logs:
            self.agent_io_logs[agent_id] = []
        
        prompt_for_log = deepcopy(user_prompt)
        prompt_for_log.pop("images_data", None)
        
        log_entry = {
            "round": self.current_round_number,
            "timestamp": datetime.datetime.now().isoformat(),
            "input": { "system_prompt": system_prompt, "user_prompt": prompt_for_log },
            "output": output
        }
        self.agent_io_logs[agent_id].append(log_entry)

    def _encode_image_to_base64(self, image_path: str) -> str:

        try:
            full_path = os.path.join("dataset/MedXpertQA/images", image_path)
            with open(full_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"[WARNING] Image file not found: {image_path}. Skipping image.")
            return None

    def _load_prompts(self, prompts_dir: str) -> dict:

        templates = {}
        required_files = [
            'student_analysis.txt', 'student_action.txt', 'patient_runtime.txt',
            'teacher_guidance.txt', 'teacher_revision.txt', 'expert_main.txt', 'supervisor_review.txt'
        ]
        try:
            for filename in required_files:
                template_name = filename.split('.')[0]
                with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                    templates[template_name] = f.read()
            return templates
        except FileNotFoundError as e:
            raise RuntimeError(f"Required prompt file not found: {e}. Please check your prompts directory.")

    def log_dialogue(self, speaker: str, content: str, visibility: str = "public"):

        turn = {"speaker": speaker, "content": content, "visibility": visibility}
        round_key = f"round_{self.current_round_number}"
        if round_key not in self.dialogue_history:
            self.dialogue_history[round_key] = []
        self.dialogue_history[round_key].append(turn)
        log_prefix = "[LOG]" if visibility == "public" else "[LOG - Internal]"
            
    def get_log_view_for(self, role: str) -> list:

        flat_history = []
        sorted_rounds = sorted(self.dialogue_history.keys())
        for round_key in sorted_rounds:
            flat_history.extend(self.dialogue_history[round_key])
        if role in 'student':
            return [turn for turn in flat_history if turn["visibility"] in ["student", "student&patient", "teacher&student"]]
        elif role == 'patient':
            return [turn for turn in flat_history if turn["visibility"] in ["patient", "student&patient"]]  
        elif role == 'teacher':
            return [turn for turn in flat_history if turn["visibility"] in ["teacher", "teacher&student", "student&patient", "teacher_private"]]
        return []

    def _save_history_to_file(self, history: dict):

        full_log = {
            "simulation_setup": {
                "selected_case": self.case_data,
                "patient_persona": self.persona,
                "selected_students": self.students
            },
            "dialogue_history": history
        }
        case_id = self.case_data.get('id', 'UnknownCase')
        filename = os.path.join(self.output_dir, f"simulation_log_{case_id}_main.json")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(full_log, f, indent=4, ensure_ascii=False)
            print(f"\n[INFO] Full simulation log successfully saved to: {filename}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save simulation log: {e}")

    def _save_agent_io_logs(self):
        print("\n[INFO] Saving individual agent I/O logs...")
        case_id = self.case_data.get('id', 'UnknownCase')

        for agent_id, logs in self.agent_io_logs.items():
            filename = os.path.join(self.output_dir, f"simulation_log_{case_id}_{agent_id}_io.json")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, indent=4, ensure_ascii=False)
                print(f"[INFO] Successfully saved I/O log for {agent_id} to: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save I/O log for {agent_id}: {e}")
    
    def run_analysis_phase(self) -> list:
        print("\n>>> Phase 1: Analysis & Reporting <<<")
        self.log_dialogue("Patient", self.current_patient_statement, visibility="student&patient")
        
        system_prompt = self.prompt_templates['student_analysis']
        unprocessed_analyses = [] 


        with ThreadPoolExecutor(max_workers=len(self.students)) as executor:

            future_to_student = {}
            for student_profile in self.students:
                user_prompt_data = {
                    "student_personal_profile": student_profile,
                    "case_summary": self.case_data.get("patient_script", {}).get("metadata", {}).get("case_title", "N/A"),
                    "dialogue_history": self.get_log_view_for('student'),
                    "patient_latest_statement": self.current_patient_statement
                }

                images_data = []
                if self.case_data.get("images"):
                    for img_filename in self.case_data["images"]:
                        encoded_img = self._encode_image_to_base64(img_filename)
                        if encoded_img:
                            images_data.append(encoded_img)
                if images_data:
                    user_prompt_data["images_data"] = images_data
                
                future = executor.submit(self.student_ai.generate, system_prompt, user_prompt_data)
                

                future_to_student[future] = (student_profile, deepcopy(user_prompt_data))

            print(f"[INFO] Waiting for {len(self.students)} students to complete their analysis in parallel...")
            for future in as_completed(future_to_student):
                student_profile, user_prompt_copy = future_to_student[future]
                student_id = student_profile["student_id"]
                try:

                    response_json = future.result()
                    
                    self.log_agent_io(f"Student_{student_id}", system_prompt, user_prompt_copy, response_json)
                    
                    analysis = response_json.get("analysis_for_teacher", f"Error: {student_id} failed to generate analysis.")
                    unprocessed_analyses.append({"student_id": student_id, "analysis": analysis})

                except Exception as exc:
                    print(f"\n[ERROR] Student {student_id} analysis generated an exception: {exc}")
                    unprocessed_analyses.append({"student_id": student_id, "analysis": f"Error during generation: {exc}"})
        
        speaking_order = random.sample([p['student_id'] for p in self.students], len(self.students))
        
        sorted_analyses = sorted(unprocessed_analyses, key=lambda x: speaking_order.index(x['student_id']))

        print("[INFO] All student analyses have been received. Logging dialogue in speaking order.")
        for analysis_item in sorted_analyses:
            self.log_dialogue(analysis_item['student_id'], analysis_item['analysis'], visibility="teacher&student")
            
        return sorted_analyses

    def run_guidance_phase(self, student_analyses: list) -> str:
        print("\n>>> Phase 2: Teacher Guidance & Review <<<")
        
        system_prompt = self.prompt_templates['teacher_guidance']
        case_data_for_teacher = {
            "case_question": self.case_data.get("question", {}),
            "case_question_answer": self.case_data.get("options", {}).get(self.case_data.get("label", {})),
            "case_images": self.case_data.get("images", {}),
            "case_body_system": self.case_data.get("body_system", {})
        }
        
        teacher_input_data = {
            "static_context": {
                "case_data": case_data_for_teacher,
                "case_socratic_steps": self.case_data.get("question_steps", [])
            },
            "dynamic_context": {
                "dialogue_history": self.get_log_view_for('teacher'),
                "current_student_analyses": student_analyses
            }
        }

        images_data = []
        if self.case_data.get("images"):
            for img_filename in self.case_data["images"]:
                encoded_img = self._encode_image_to_base64(img_filename)
                if encoded_img:
                    images_data.append(encoded_img)
        if images_data:
            teacher_input_data["images_data"] = images_data
        
        is_approved = False
        teacher_draft = ""
        max_retries = 3
        
        for i in range(max_retries):
            user_prompt_copy = deepcopy(teacher_input_data)
            response_json = self.teacher_ai.generate(system_prompt, teacher_input_data)
            self.log_agent_io("Teacher", system_prompt, user_prompt_copy, response_json)
            
            if "revised_guidance" in response_json:
                teacher_draft = response_json.get("revised_guidance")
            else:
                teacher_draft = response_json.get("guidance")

            if not teacher_draft:
                 teacher_draft = f"Error: Teacher failed to generate on attempt {i+1}."
            
            self.log_dialogue(f"Teacher (Draft {i+1})", teacher_draft, visibility="teacher")

            expert_feedback = None
            supervisor_feedback = None

            print("[INFO] Starting Expert and Supervisor reviews in parallel...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                expert_prompt = self.prompt_templates['expert_main']
                expert_input = { "mode": "fact_check", "case_data": case_data_for_teacher, "teacher_statement": teacher_draft }
                if images_data: expert_input["images_data"] = images_data
                expert_future = executor.submit(self.expert_ai.generate, expert_prompt, expert_input)

                supervisor_prompt = self.prompt_templates['supervisor_review']
                supervisor_input = {"teacher_statement": teacher_draft}
                supervisor_future = executor.submit(self.supervisor_ai.generate, supervisor_prompt, supervisor_input)

                try:
                    expert_review = expert_future.result()
                    self.log_agent_io("Expert", expert_prompt, deepcopy(expert_input), expert_review)
                except Exception as exc:
                    print(f"\n[ERROR] Expert review generated an exception: {exc}")
                    expert_review = {"is_correct": False, "feedback": f"Error during generation: {exc}"}

                try:
                    supervisor_review = supervisor_future.result()
                    self.log_agent_io("Supervisor", supervisor_prompt, supervisor_input, supervisor_review)
                except Exception as exc:
                    print(f"\n[ERROR] Supervisor review generated an exception: {exc}")
                    supervisor_review = {"is_safe": False, "feedback_and_suggestion": f"Error during generation: {exc}"}
            
            print("[INFO] Both reviews completed.")
        
            if not expert_review.get("is_correct", True):
                expert_feedback = expert_review.get('feedback', 'Fact check failed.')
                self.log_dialogue("Expert (Review)", f"FAILED: {expert_feedback}", visibility="teacher_private")
            
            if not supervisor_review.get("is_safe", True):
                supervisor_feedback = supervisor_review.get('feedback_and_suggestion', 'Safety check failed.')
                self.log_dialogue("Supervisor (Review)", f"FAILED: {supervisor_feedback}", visibility="teacher_private")


            if expert_feedback is None and supervisor_feedback is None:
                is_approved = True
                print("Guidance approved by all reviewers.")
                self.log_dialogue("Expert (Review)", "Fact check passed. The guidance is factually correct.", visibility="teacher_private")
                self.log_dialogue("Supervisor (Review)", "Safety check passed. The guidance is safe and ethical.", visibility="teacher_private")
                break
            else:
                print(f"Review failed on attempt {i+1}. Switching to revision mode.")
                system_prompt = self.prompt_templates['teacher_revision']
                teacher_input_data = {
                    "previous_guidance": teacher_draft,
                    "feedback": {
                        "Medical_Knowledge_Expert": expert_feedback, 
                        "Safety_Ethics_Supervisor": supervisor_feedback
                    },
                    "context": teacher_input_data.get("context", teacher_input_data)
                }
        
        if not is_approved:
            teacher_draft = "Let's pause and reconsider. What is the most critical piece of information we need right now?"
            self.log_dialogue("System", f"Teacher draft failed all reviews. Using a fallback guidance.", visibility="teacher")

        self.log_dialogue("Teacher", teacher_draft, visibility="teacher&student")
        return teacher_draft

    def run_query_phase(self, teacher_guidance: str) -> str:
        print("\n>>> Phase 3: Query & Exploration <<<")
        
        system_prompt = self.prompt_templates['student_action']
        student_actions = [] 

        with ThreadPoolExecutor(max_workers=len(self.students)) as executor:
            future_to_student = {}
            for student_profile in self.students:
                user_prompt_data = {
                    "student_personal_profile": student_profile,
                    "case_summary": self.case_data.get("patient_script", {}).get("metadata", {}).get("case_title", "N/A"),
                    "dialogue_history": self.get_log_view_for('student'),
                    "teacher_latest_guidance": teacher_guidance
                }
                future = executor.submit(self.student_ai.generate, system_prompt, user_prompt_data)
                future_to_student[future] = (student_profile, deepcopy(user_prompt_data))

            print(f"[INFO] Waiting for {len(self.students)} students to formulate their actions in parallel...")
            for future in as_completed(future_to_student):
                student_profile, user_prompt_copy = future_to_student[future]
                student_id = student_profile["student_id"]
                try:
                    response_json = future.result()
                    self.log_agent_io(f"Student_{student_id}", system_prompt, user_prompt_copy, response_json)
                    student_actions.append({"student_id": student_id, "action": response_json})
                except Exception as exc:
                    print(f"\n[ERROR] Student {student_id} action generation generated an exception: {exc}")
                    student_actions.append({"student_id": student_id, "action": {"error": f"Error: {exc}"}})
        
        all_patient_queries = []
        expert_query_tasks = []
        
        speaking_order = random.sample([p['student_id'] for p in self.students], len(self.students))
        sorted_actions = sorted(student_actions, key=lambda x: speaking_order.index(x['student_id']))

        print("[INFO] All student actions received. Logging queries and preparing expert Q&A.")
        for action_item in sorted_actions:
            student_id = action_item['student_id']
            action = action_item['action']
            
            if action.get("query_for_patient"):
                query = action["query_for_patient"]
                self.log_dialogue(student_id, f"(query_for_patient) {query}", visibility="student&patient")
                all_patient_queries.append(query)
            
            if action.get("query_for_expert"):
                query = action["query_for_expert"]
                self.log_dialogue(student_id, f"(query_for_expert) {query}", visibility="student")
                expert_query_tasks.append({"student_id": student_id, "query": query})
        
        if expert_query_tasks:
            print(f"[INFO] Processing {len(expert_query_tasks)} expert queries in parallel...")
            expert_prompt = self.prompt_templates['expert_main']
            with ThreadPoolExecutor(max_workers=len(expert_query_tasks)) as executor:
                future_to_expert_query = {}
                for task in expert_query_tasks:
                    expert_input = {"mode": "knowledge_query", "student_statement": task['query']}
                    future = executor.submit(self.expert_ai.generate, expert_prompt, expert_input)
                    future_to_expert_query[future] = task

                for future in as_completed(future_to_expert_query):
                    task_info = future_to_expert_query[future]
                    try:
                        expert_answer = future.result()
                        self.log_agent_io("Expert", expert_prompt, {"mode": "knowledge_query", "student_statement": task_info['query']}, expert_answer)
                        if expert_answer.get("answer_provided"):
                            self.log_dialogue("Medical Expert", expert_answer.get("explanation"), visibility="student")
                    except Exception as exc:
                        print(f"\n[ERROR] Expert query from {task_info['student_id']} generated an exception: {exc}")
                        self.log_dialogue("Medical Expert", f"Sorry, I encountered an error trying to answer the question: '{task_info['query']}'", visibility="student")
            print("[INFO] All expert queries have been answered.")

        if not all_patient_queries:
            print("[INFO] No questions for the patient in this round.")
            return "No question from students"

        print("[INFO] Compiling all questions for the patient...")
        patient_system_prompt = self.prompt_templates['patient_runtime']
        patient_user_data = {
            "script": { "persona": self.persona, "case_facts": self.case_data.get("patient_script", {}) },
            "dialogue_history": self.get_log_view_for('patient'),
            "student_queries": all_patient_queries
        }
        
        patient_user_copy = deepcopy(patient_user_data)
        patient_response = self.patient_ai.generate(patient_system_prompt, patient_user_data)
        self.log_agent_io("Patient", patient_system_prompt, patient_user_copy, patient_response)

        new_patient_statement = patient_response.get("response", "I'm not sure how to answer that.")
        return new_patient_statement

    def run_simulation(self, max_rounds=3):
        for i in range(max_rounds):
            self.current_round_number = i + 1
            print(f"\n{'='*20} ROUND {self.current_round_number} {'='*20}")
            
            student_analyses = self.run_analysis_phase()
            teacher_guidance = self.run_guidance_phase(student_analyses)
            new_patient_statement = self.run_query_phase(teacher_guidance)
            
            self.current_patient_statement = new_patient_statement
            
        print(f"\n{'='*20} SIMULATION END {'='*20}")
        
        self._save_history_to_file(self.dialogue_history)
        self._save_agent_io_logs()

        return self.dialogue_history

def setup_simulation_for_case(case: dict, persona_library: list, student_library: list):
    print(f"--- Setting up simulation data for Case ID: {case['id']} ---")
    
    patient_script = case.get('patient_script', {})
    final_persona = None
    num_students = random.choice([1,2,3,4])

    case_metadata = patient_script.get("metadata", {})
    if "demographics" in case_metadata:
        print("Case has specific demographics. Applying hard matching rules...")
        case_demographics = case_metadata["demographics"]
        target_gender = case_demographics.get("gender")
        target_age = case_demographics.get("age")

        if not target_gender or target_age is None:
            raise ValueError(f"Case {case['id']} has incomplete demographics.")

        gender_matches = [p for p in persona_library if p.get("demographics", {}).get("gender") == target_gender]
        if not gender_matches:
            raise ValueError(f"No personas found with matching gender: {target_gender}")
        
        print(f"Found {len(gender_matches)} personas with matching gender '{target_gender}'.")

        sorted_matches = sorted(gender_matches, key=lambda p: abs(p.get("demographics", {}).get("age", 999) - target_age))
        
        top_candidates = sorted_matches[:4]
        print(f"Found {len(top_candidates)} closest age candidates for random selection.")

        selected_persona = random.choice(top_candidates)
        
        final_persona = deepcopy(selected_persona)
        final_persona["demographics"] = case_demographics
        print(f"Randomly selected Persona '{final_persona['persona_id']}' from top candidates. Demographics finalized from case data.")
    else:
        print("Case is generic. Selecting a completely random persona.")
        final_persona = deepcopy(random.choice(persona_library))
        print(f"Randomly selected persona: {final_persona['persona_id']}")

    unique_student_ids_in_library = {s['student_id'] for s in student_library}
    if len(unique_student_ids_in_library) < num_students:
        raise ValueError(f"Cannot select {num_students} unique students, because only {len(unique_student_ids_in_library)} unique student IDs are available in the library.")

    selected_students = []
    selected_ids = set()
    while len(selected_students) < num_students:
        candidate_student = random.choice(student_library)
        candidate_id = candidate_student['student_id']
        if candidate_id not in selected_ids:
            selected_students.append(candidate_student)
            selected_ids.add(candidate_id)
            
    final_student_ids = [s["student_id"] for s in selected_students]
    print(f"Selected Unique Students: {final_student_ids}")
    
    print("--- Simulation data setup complete. ---")
    return final_persona, selected_students


if __name__ == '__main__':
    PERSONA_FILE_PATH = "Our_framework/Patient_simulate/patient_personas.json"
    STUDENT_FILE_PATH = "Our_framework/Student_simulate/student_personas.json"
    PROMPTS_DIR = "Our_framework/prompt/Agent"
    CASE_FILE_PATH = "Our_framework/Patient_simulate/MedXpert_patient_script.json"
    
    AI_CONFIG = {
        'api_key': "", 
        'base_url': '',
        'models': { 'student': 'gpt-4o', 'teacher': 'gpt-4o', 'patient': 'gpt-4.1', 'expert': 'gemini-2.5-flash', 'supervisor': 'gemini-2.5-flash' }
    }

    TOTAL_CHUNKS = 7
    CURRENT_CHUNK_INDEX = 1 

    if not 1 <= CURRENT_CHUNK_INDEX <= TOTAL_CHUNKS:
        raise ValueError(f"CURRENT_CHUNK_INDEX must be between 1 and {TOTAL_CHUNKS}")

    BASE_SAVE_DIR = "Our_framework/Simulation_log/MM_test" 
    CHUNK_SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"MM_test_Chunk_{CURRENT_CHUNK_INDEX}_of_{TOTAL_CHUNKS}")

    PROGRESS_FILE_PATH = os.path.join(BASE_SAVE_DIR, f"MM_test_simulation_progress_chunk_{CURRENT_CHUNK_INDEX}.log")
    
    os.makedirs(CHUNK_SAVE_DIR, exist_ok=True) 

    try:
        print("Loading data libraries...")
        with open(CASE_FILE_PATH, 'r', encoding='utf-8') as f:
            case_library = json.load(f)
        with open(PERSONA_FILE_PATH, 'r', encoding='utf-8') as f:
            persona_library = json.load(f)
        with open(STUDENT_FILE_PATH, 'r', encoding='utf-8') as f:
            student_library = json.load(f)
        print(f"\nAll data libraries loaded successfully. Found {len(case_library)} total cases.")

        total_cases = len(case_library)
        chunk_size = (total_cases + TOTAL_CHUNKS - 1) // TOTAL_CHUNKS
        start_index = (CURRENT_CHUNK_INDEX - 1) * chunk_size
        end_index = min(start_index + chunk_size, total_cases)
        cases_to_process = case_library[start_index:end_index]
        
        print(f"\n--- Running chunk {CURRENT_CHUNK_INDEX}/{TOTAL_CHUNKS} ---")
        print(f"Saving logs to: {CHUNK_SAVE_DIR}")
        print(f"Processing cases from index {start_index} to {end_index - 1} (total {len(cases_to_process)} cases).")

        completed_cases = set()
        if os.path.exists(PROGRESS_FILE_PATH):
            with open(PROGRESS_FILE_PATH, 'r', encoding='utf-8') as f:
                completed_cases = {line.strip() for line in f if line.strip()}
            print(f"[RESUME INFO] Loaded {len(completed_cases)} completed case IDs for this chunk.")
        else:
            print("[RESUME INFO] Progress file for this chunk not found. Starting new.")

        for i, current_case in enumerate(cases_to_process):
            global_index = start_index + i
            case_id = current_case.get('id', 'UnknownCase')
            
            print(f"\n{'='*40}")
            print(f"Processing Case {global_index + 1}/{total_cases}: ID = {case_id}")
            print(f"{'='*40}")
            
            if case_id in completed_cases:
                print(f"[RESUME] Case ID {case_id} is already completed. Skipping.")
                continue
            
            try:
                final_persona, final_students = setup_simulation_for_case(
                    current_case, persona_library, student_library
                )

                orchestrator = Orchestrator(
                    case_data=current_case,
                    persona=final_persona,
                    student_profiles=final_students,
                    ai_config=AI_CONFIG,
                    prompts_dir=PROMPTS_DIR,
                    save_dir=CHUNK_SAVE_DIR  
                )

                orchestrator.run_simulation(max_rounds=random.choice([3,4,5]))

                with open(PROGRESS_FILE_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{case_id}\n")
                completed_cases.add(case_id)
                print(f"[PROGRESS] Marked Case ID {case_id} as completed for this chunk.")

            except (ValueError, RuntimeError) as e:
                print(f"\n[CRITICAL ERROR] Failed to process case {case_id}. Error: {e}")
                print("Skipping to the next case in this chunk...")
                continue
        
        print(f"\nChunk {CURRENT_CHUNK_INDEX}/{TOTAL_CHUNKS} has been processed.")

    except FileNotFoundError as e:
        print(f"\n[SETUP FAILED] A required data file was not found: {e}")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred during setup: {e}")