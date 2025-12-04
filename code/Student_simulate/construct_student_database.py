import json
import os
import time
import re
import uuid
from openai import OpenAI


client = OpenAI(
    base_url='',
    api_key=os.getenv("OPENAI_API_KEY", '') 
)

PROMPT_FILE = "Our_framework/Student_simulate/construct_student_database.txt"
OUTPUT_FILE = "Our_framework/Student_simulate/student_personas.json"
TOTAL_PERSONAS_TO_GENERATE = 300
BATCH_SIZE = 5 


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

def generate_persona_batch(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_tokens=4000, 
        )
        
        raw_response_text = response.choices[0].message.content
        parsed_response = extract_json_from_response(raw_response_text)
        
        if isinstance(parsed_response, list):
            return parsed_response
        else:
            print("Error")
            return None
        
    except Exception as e:
        print(f"Error (API): {str(e)}")
        return None

def main():
    try:
        base_prompt = load_prompt_template(PROMPT_FILE)
    except Exception as e:
        print(f"Error '{PROMPT_FILE}'  {str(e)}")
        return

    all_personas = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    all_personas = json.loads(content)
            print(f"Loaded {len(all_personas)} roles")
        except json.JSONDecodeError:
            print(f"Error '{OUTPUT_FILE}' ")
            all_personas = []

    num_existing = len(all_personas)
    num_to_generate = TOTAL_PERSONAS_TO_GENERATE - num_existing
    
    if num_to_generate <= 0:
        print(f"Target ({TOTAL_PERSONAS_TO_GENERATE}) Done")
        return
        
    num_batches_to_run = (num_to_generate + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Target: {TOTAL_PERSONAS_TO_GENERATE}, Done: {num_existing}")
    
    for i in range(num_batches_to_run):
        current_total = len(all_personas)
        
        persona_batch = generate_persona_batch(base_prompt)
        
        if not persona_batch or not all(isinstance(p, dict) for p in persona_batch):
            print("Error")
            time.sleep(5) 
            continue
        
        for persona in persona_batch:
            persona['persona_id'] = f"persona_{uuid.uuid4()}"
            all_personas.append(persona)
        
        temp_output_file = OUTPUT_FILE + '.tmp'
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_personas, f, indent=2, ensure_ascii=False)
        os.replace(temp_output_file, OUTPUT_FILE)
        
    
        time.sleep(5) 

    print(f"\nDone {OUTPUT_FILE}")
    print(f"Total {len(all_personas)} roles。")


if __name__ == "__main__":
    main()