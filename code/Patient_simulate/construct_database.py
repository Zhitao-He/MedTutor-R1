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

PROMPT_FILE = "Our_framework/Patient_simulate/personality_database.txt"
OUTPUT_FILE = "Our_framework/Patient_simulate/patient_personas.json"
TOTAL_PERSONAS_TO_GENERATE = 300


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


def generate_single_persona(prompt):

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8, 
            max_tokens=1500, 
        )
        
        raw_response_text = response.choices[0].message.content
        
        parsed_response = extract_json_from_response(raw_response_text)
        
        return parsed_response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    try:
        base_prompt = load_prompt_template(PROMPT_FILE)
    except Exception as e:
        print(f"File '{PROMPT_FILE}' Error: {str(e)}")
        return

    all_personas = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    all_personas = json.loads(content)
        except json.JSONDecodeError:
            print(f"Error '{OUTPUT_FILE}'")
            all_personas = []

    num_existing = len(all_personas)
    num_to_generate = TOTAL_PERSONAS_TO_GENERATE - num_existing
    
    if num_to_generate <= 0:
        print(f"Target ({TOTAL_PERSONAS_TO_GENERATE}) Done")
        return
    
    print(f"Target: {TOTAL_PERSONAS_TO_GENERATE}, Done: {num_existing}, Create: {num_to_generate}个。")

    for i in range(num_to_generate):
        print(f"\n--- Creating {num_existing + i + 1}/{TOTAL_PERSONAS_TO_GENERATE} ---")
        
        persona_data = generate_single_persona(base_prompt)

        final_persona = None
        if isinstance(persona_data, dict):
            final_persona = persona_data
        elif isinstance(persona_data, list) and persona_data and isinstance(persona_data[0], dict):
            final_persona = persona_data[0]

        if final_persona is None:
            print("Error")
            continue

        final_persona['persona_id'] = f"persona_{uuid.uuid4()}"
        
        all_personas.append(final_persona)
        
        temp_output_file = OUTPUT_FILE + '.tmp'
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_personas, f, indent=2, ensure_ascii=False)
        os.replace(temp_output_file, OUTPUT_FILE)
        
        print(f"Role '{final_persona.get('demographics', {}).get('name', 'N/A')}' (ID: {final_persona['persona_id']}) Done")
        
        time.sleep(2)

    print(f"\nDone {OUTPUT_FILE}")
    print(f"{len(all_personas)} roles。")

if __name__ == "__main__":
    main()