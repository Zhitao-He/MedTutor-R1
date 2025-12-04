import os
import json
import re
import time
import random
from typing import Dict, Any, TypedDict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def log_model_output(prompt: str, response: str):
    """Appends the model's prompt and response to a log file."""
    try:
        # Use 'a' for append mode so you don't overwrite the file on each batch.
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            log_entry = (
                f"--- LOG ENTRY: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                f"[PROMPT]:\n{prompt}\n\n"
                f"[MODEL RESPONSE]:\n{response}\n\n"
                f"---------------------------------------------------\n\n"
            )
            f.write(log_entry)
    except Exception as e:
        # Print an error if logging fails, but don't crash the main process.
        print(f"!!! Failed to write to log file {LOG_FILE_PATH}: {e}")
# --- 1. API Client and Setup ---
API_KEY = os.getenv("NUWA_API_KEY", "Your_API_Keys")
if not API_KEY:
    raise ValueError("API key not found. Please set the NUWA_API_KEY environment variable or hardcode it.")
client = OpenAI(base_url='https://api.nuwaapi.com/v1', api_key=API_KEY)

def call_openai_api(prompt: str, model: str = "gpt-4.1", temperature: float = 0.2, max_tokens: int = 500, max_retries: int = 3) -> Optional[str]:
    # ... (your API call function) ...
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0, 0.5))
            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens, timeout=90)
            return response.choices[0].message.content.strip()
        except Exception as e:
            prompt_snippet = prompt[:100].replace('\n', ' ')
            print(f"API call failed for prompt '{prompt_snippet}...'(attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(5 * (attempt + 1))
    print(f"API call failed after {max_retries} attempts. Returning None.")
    return None

class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str

class RewardScore(TypedDict):
    overall: float
    r_base: float
    i_veto: int
    CS_1: Optional[float]; CS_2: Optional[float]; SGA_1: Optional[float]; SGA_2: Optional[float]; SGA_3: Optional[float]; IS_1: Optional[float]; IS_2: Optional[float]; IS_3: Optional[float]

# --- 2. Prompt Formatting ---
def format_analysis_prompt(full_input_prompt: str, model_output: str) -> str:
    """
    Formats a new prompt for the Analysis Quality Judge that asks for a 
    binary (0 or 1) score.
    """
    return f"""
Role
You are a seasoned medical educator and a clinical reasoning analyst. Your task is to perform a holistic assessment of an AI teacher's analysis quality.

Task
Based on the [Evaluation Criteria] below, you must determine if the provided [Model Output] demonstrates high-quality analysis. You must assign a single overall score: **1 for high quality, and 0 for low quality**. Provide a brief justification for your decision.

Evaluation Context
{full_input_prompt}

Model Output to Evaluate
{model_output}

Evaluation Criteria (Consider these aspects for your overall judgment)
SGA-1 (Individual Assessment): Is the analysis for each student within the <think_student> tags accurate and deep?
SGA-2 (Group Synthesis): Does the <think_group> tag accurately capture the team's dynamics and collective cognitive state?
SGA-3 (Image Correlation): Is the analysis of medical imagery in the <think_image> tag accurate and relevant?

Output Format
You must strictly return your evaluation results in the following JSON format, with no other text or explanation:
{{
"quality_score": <0_or_1>,
"reason": "<A brief justification for your overall 0 or 1 score>"
}}
"""

# --- 3. Rule-Based and Parsing Logic (Helpers for both reward functions) ---
def check_structural_integrity(response: str) -> bool:
    patterns = [r"<think_history>", r"<think_question>", r"<think_student[ >]", r"<think_group>", r"<think_image>"]
    return all(re.search(pattern, response) for pattern in patterns)

def check_student_consistency(prompt: str, response: str) -> bool:
    try:
        analysis_section = prompt.split("Current Student Analyses")[1]
        student_names = re.findall(r"-\s*(\w+):", analysis_section)
    except IndexError:
        student_names = []
    student_blocks = re.findall(r"(<think_student.*?>.*?</think_student>)", response, re.DOTALL)
    if len(student_names) != len(student_blocks): return False
    if not student_names: return True
    all_blocks_text = "".join(student_blocks)
    return all(re.search(r'\b' + re.escape(name) + r'\b', all_blocks_text, re.IGNORECASE) for name in student_names)

def get_student_consistency_counts(prompt: str, response: str) -> tuple[int, int]:
    # ... (your get_student_consistency_counts function) ...
    try:
        analysis_section = prompt.split("Current Student Analyses")[1]
        student_names_from_prompt = re.findall(r"-\s*(\w+):", analysis_section)
    except IndexError:
        student_names_from_prompt = []
    total_count = len(student_names_from_prompt)
    if total_count == 0: return 0, 0
    student_blocks = re.findall(r"(<think_student.*?>.*?</think_student>)", response, re.DOTALL)
    all_blocks_text = "".join(student_blocks)
    found_count = sum(1 for name in student_names_from_prompt if re.search(r'\b' + re.escape(name) + r'\b', all_blocks_text, re.IGNORECASE))
    return found_count, total_count


def _calculate_single_vanilla_reward(reward_input: RewardInput) -> Dict[str, float]:
    """
    Helper function containing the logic to process ONE sample.
    This will be run in parallel by the main batch function.
    """
    prompt = reward_input["ground_truth"]
    response = reward_input["response"]

    # --- Step 1: Calculate Rule-Based Format Score ---
    is_structurally_correct = check_structural_integrity(response)
    is_student_consistent = check_student_consistency(prompt, response)
    format_score = 1.0 if is_structurally_correct and is_student_consistent else 0.0
    
    # --- Step 2: Calculate LLM-Based Content Score ---
    analysis_prompt = format_analysis_prompt(prompt, response)
    analysis_response_str = call_openai_api(analysis_prompt)
    print(f"👉 Format Score was: {format_score}")
    print(f"📞 Raw response from Judge API call:\n'''\n{analysis_response_str}\n'''")
    
    content_score = 0.0
    if analysis_response_str:
        try:
            match = re.search(r'\{.*\}', analysis_response_str, re.DOTALL)
            if match:
                judge_json = json.loads(match.group(0))
                content_score = float(judge_json.get("quality_score", 0.0))
                print(f"✅ Successfully parsed Judge JSON. Content Score: {content_score}")
        except (json.JSONDecodeError, TypeError):
            content_score = 0.0
    
    # --- Step 3: Calculate Final Reward and Return ---
    overall_reward = format_score + content_score

    return {
        "overall": overall_reward,
        "format_score": format_score,
        "content_score": content_score
    }

# --- 6. Main BATCH Function (Entry point for the trainer) ---

def calculate_vanilla_reward_batch(
    reward_inputs: List[Dict[str, Any]], 
    max_workers: int = 16  # Concurrency for API calls
) -> List[Dict[str, float]]:
    """
    This is the new entry point for batch processing.
    It takes a list of samples and uses a ThreadPoolExecutor to call 
    the single-sample reward function in parallel.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("This function is designed for `reward_type=batch` and expects a list of inputs.")

    print(f"📝 Logging batch of {len(reward_inputs)} responses to {LOG_FILE_PATH}...")
    for item in reward_inputs:
        # The 'response' key holds the model's generated text.
        # The 'ground_truth' key holds the original prompt.
        log_model_output(prompt=item.get("ground_truth", "PROMPT NOT FOUND"), 
                         response=item.get("response", "RESPONSE NOT FOUND"))
                         
    print(f"🚀 Received a batch of {len(reward_inputs)} items. Processing in parallel with up to {max_workers} workers...")
    
    # Pre-allocate list to store results in the correct order
    results = [None] * len(reward_inputs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping from future to index to re-order the results
        future_to_index = {
            executor.submit(_calculate_single_vanilla_reward, item): i 
            for i, item in enumerate(reward_inputs)
        }
        
        # as_completed yields futures as they finish, which is efficient
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                print(f"Item at index {index} generated an exception during processing: {exc}")
                # On error, assign a default penalty score to avoid crashing
                results[index] = {"overall": 0.0, "format_score": 0.0, "content_score": 0.0}

    print(f"✅ Finished processing batch of {len(reward_inputs)} items.")
    
    # Filter out any potential None results just in case, though the exception handling should prevent this
    final_results = [res for res in results if res is not None]
    if len(final_results) != len(reward_inputs):
        print(f"⚠️ Warning: Some items failed to process. Expected {len(reward_inputs)} results, but got {len(final_results)}.")

    return final_results