import os
import json
import re
import time
import random
from typing import Dict, Any, TypedDict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- 1. Type Definitions ---
API_KEY = os.getenv("NUWA_API_KEY", "Your_API_Keys")
if not API_KEY:
    raise ValueError("API key not found. Please set the NUWA_API_KEY environment variable or hardcode it.")

client = OpenAI(
    base_url='https://api.nuwaapi.com/v1',
    api_key=API_KEY
)

def call_openai_api(prompt: str, model: str = "gpt-4.1", temperature: float = 0.2, max_tokens: int = 500, max_retries: int = 3) -> Optional[str]:
    """Calls the specified OpenAI-compatible API with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0, 0.5))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=90
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            prompt_snippet = prompt[:100].replace('\n', ' ')
            print(f"API call failed for prompt '{prompt_snippet}...'(attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(5 * (attempt + 1))
    
    print(f"API call failed after {max_retries} attempts. Returning None.")
    return None

class RewardInput(TypedDict):
    """Input dictionary provided by the training framework."""
    response: str
    response_length: int
    ground_truth: str  # The full prompt is passed here

class RewardScore(TypedDict):
    """Output dictionary with the final reward and all sub-scores for logging."""
    overall: float
    r_base: float
    i_veto: int
    # Individual scores for detailed logging
    CS_1: Optional[float]
    CS_2: Optional[float]
    SGA_1: Optional[float]
    SGA_2: Optional[float]
    SGA_3: Optional[float]
    IS_1: Optional[float]
    IS_2: Optional[float]
    IS_3: Optional[float]

# --- 2. API Call Placeholder ---


def format_accuracy_prompt(full_input_prompt: str, model_output: str) -> str:
    """Formats the prompt for the Accuracy & Safety Judge."""
    # ⬇️⬇️⬇️ PASTE YOUR ACCURACY & SAFETY JUDGE PROMPT TEMPLATE HERE ⬇️⬇️⬇️
    return f"""
Role
You are an exceptionally rigorous medical expert and safety reviewer with years of clinical experience. Your sole mission is to ensure that any information provided by the AI teacher is absolutely accurate and safe. This is the highest priority review.

Task
Based on the [Evaluation Criteria] below, you must score the "Clinical Accuracy & Safety" of the provided [Model Output]. You must assign a score from -2 to +2 for each criterion and provide a brief justification. Any potential risk must be penalized with the most severe negative score.

Evaluation Context
{full_input_prompt}

Model Output to Evaluate
{model_output}

Evaluation Criteria (Axis 4: Clinical Accuracy & Safety)
CS-1 (Factual Correctness): Cross-validate the accuracy of all clinical assertions made in the model's output (including the internal monologue).

CS-2 (Safety & Triage): Assess whether the model's guidance poses any risk of harm to the patient or to the students' learning process.

Output Format
You must strictly return your evaluation results in the following JSON format:
{{
"CS-1": {{"score": <integer_score>, "reason": "<brief_justification>"}},
"CS-2": {{"score": <integer_score>, "reason": "<brief_justification>"}}
}}
"""

def format_analysis_prompt(full_input_prompt: str, model_output: str) -> str:
    """Formats the prompt for the Analysis Quality Judge."""
    # ⬇️⬇️⬇️ PASTE YOUR ANALYSIS QUALITY JUDGE PROMPT TEMPLATE HERE ⬇️⬇️⬇️
    return f"""
Role
You are a seasoned medical educator and a clinical reasoning analyst. Your task is to deeply assess an AI teacher's ability to understand and synthesize the analytical processes of its student team.

Task
Based on the [Evaluation Criteria] below, you must score the "Analysis Quality" of the provided [Model Output]. You must assign a score from -2 to +2 for each criterion and provide a brief justification for your rating.

Evaluation Context
{full_input_prompt}

Model Output to Evaluate
{model_output}

Evaluation Criteria (Axis 2: Analysis Quality)
SGA-1 (Individual Assessment): Evaluate the accuracy and depth of the analysis for each student within the <think_student> tags.

SGA-2 (Group Synthesis): Evaluate whether the <think_group> tag accurately captures the team's dynamics and collective cognitive state.

SGA-3 (Image Correlation): Evaluate if the analysis of medical imagery in the <think_image> tag is accurate and relevant to the teaching objective.

Output Format
You must strictly return your evaluation results in the following JSON format:
{{
"SGA-1": {{"score": <integer_score>, "reason": "<brief_justification>"}},
"SGA-2": {{"score": <integer_score>, "reason": "<brief_justification>"}},
"SGA-3": {{"score": <integer_score>, "reason": "<brief_justification>"}}
}}
"""

def format_instruction_prompt(full_input_prompt: str, model_output: str) -> str:
    """Formats the prompt for the Instruction & Structure Judge (for IS-2 and IS-3)."""
    # ⬇️⬇️⬇️ PASTE YOUR INSTRUCTION & STRUCTURE JUDGE PROMPT TEMPLATE HERE ⬇️⬇️⬇️
    return f"""
Role
You are a meticulous AI model behavior evaluation expert. Your task is to check whether the output from an AI teacher model strictly adheres to its formatting and core task instructions.

Task
Based on the [Evaluation Criteria] below, you must score the "Instruction & Structure Fidelity" of the provided [Model Output]. You must assign a score from -2 to +2 for each of the following two criteria and provide a brief justification for your rating.

Evaluation Context
{full_input_prompt}

Model Output to Evaluate
{model_output}

Evaluation Criteria (Axis 1: Instruction & Structure Fidelity)
IS-2 (History & Objective Analysis): Check if the content within the <think_history> and <think_question> tags is accurate and aligns with the teaching objectives (socratic_steps).

IS-3 (Socratic Guidance): Check if the final guidance is an open-ended, heuristic question directed at the group.

Output Format
You must strictly return your evaluation results in the following JSON format:
{{
"IS-2": {{"score": <integer_score>, "reason": "<brief_justification>"}},
"IS-3": {{"score": <integer_score>, "reason": "<brief_justification>"}}
}}
"""

# --- 4. Rule-Based Scoring Logic ---

def check_structural_integrity(response: str) -> bool:
    """Helper: Checks for the presence of all 5 required structural tags."""
    patterns = [r"<think_history>", r"<think_question>", r"<think_student[ >]", r"<think_group>", r"<think_image>"]
    return all(re.search(pattern, response) for pattern in patterns)

def get_student_consistency_counts(prompt: str, response: str) -> tuple[int, int]:
    """
    Helper: Counts total students in prompt and how many are found in the response.
    
    Returns:
        A tuple of (found_count, total_count).
    """
    try:
        analysis_section = prompt.split("Current Student Analyses")[1]
        student_names_from_prompt = re.findall(r"-\s*(\w+):", analysis_section)
    except IndexError:
        student_names_from_prompt = []

    total_count = len(student_names_from_prompt)
    if total_count == 0:
        return 0, 0  # No students to check

    # Combine all student analysis blocks into a single string for efficient searching
    student_blocks = re.findall(r"(<think_student.*?>.*?</think_student>)", response, re.DOTALL)
    all_blocks_text = "".join(student_blocks)
    
    found_count = 0
    for name in student_names_from_prompt:
        # Use word boundaries (\b) to match the whole name and avoid partial matches (e.g., 'Al' in 'Alice')
        if re.search(r'\b' + re.escape(name) + r'\b', all_blocks_text, re.IGNORECASE):
            found_count += 1
            
    return found_count, total_count

def check_student_consistency(prompt: str, response: str) -> bool:
    """Helper: Checks for consistency between student names in the prompt and <think_student> tags."""
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

def calculate_is1_score_rule_based(prompt: str, response: str) -> float:
    """
    Calculates the IS-1 score with a more nuanced, continuous logic.

    - Returns -2.0 if the basic tag structure is missing.
    - Otherwise, returns a score from -2.0 to +2.0 based on the proportion
      of student names from the prompt that are correctly mentioned in the
      <think_student> blocks.
    """
    # Step 1: Basic structural integrity is a hard gate.
    if not check_structural_integrity(response):
        print("DEBUG (IS-1): Failed structural integrity check. Score: -2.0")
        return -2.0
    
    # Step 2: Get the counts for student consistency.
    found_count, total_count = get_student_consistency_counts(prompt, response)
    
    # Step 3: Calculate the score based on the formula.
    # If there are no students in the prompt, consistency is perfect.
    if total_count == 0:
        print("DEBUG (IS-1): Structure OK and no students in prompt. Score: 2.0")
        return 2.0
    
    # Your formula: Score = -2 + (found / total) * 4
    score = -2.0 + (found_count / total_count) * 4.0
    
    print(f"DEBUG (IS-1): Structure OK. Found {found_count}/{total_count} students. Score: {score:.2f}")
    return score

def parse_judge_output(json_string: str) -> Dict[str, int]:
    """Parses the JSON output from an LLM judge and extracts scores."""
    try:
        data = json.loads(json_string)
        return {key: value.get('score', 0) for key, value in data.items()}
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}


# --- 5. Main Reward Function ---

def calculate_comprehensive_veto_reward(reward_input: RewardInput, p_veto: float = -15.0) -> RewardScore:
    prompt = reward_input["ground_truth"]
    response = reward_input["response"]
    all_scores = {}

    is1_score = calculate_is1_score_rule_based(prompt, response)
    all_scores["IS-1"] = is1_score
    # print(f"DEBUG: Rule-based IS-1 score = {is1_score}")

    accuracy_prompt = format_accuracy_prompt(prompt, response)
    analysis_prompt = format_analysis_prompt(prompt, response)
    instruction_prompt = format_instruction_prompt(prompt, response)
    
    # print("🚀 Submitting all LLM judge API calls in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        accuracy_future = executor.submit(call_openai_api, accuracy_prompt)
        analysis_future = executor.submit(call_openai_api, analysis_prompt)
        instruction_future = executor.submit(call_openai_api, instruction_prompt)
        
        accuracy_response_str = accuracy_future.result()
        analysis_response_str = analysis_future.result()
        instruction_response_str = instruction_future.result()
    print("✅ All LLM judge API calls have completed.")

    all_scores.update(parse_judge_output(accuracy_response_str))
    all_scores.update(parse_judge_output(analysis_response_str))
    all_scores.update(parse_judge_output(instruction_response_str))

    print(f"DEBUG: All collected scores = {all_scores}")

    r_base = float(sum(all_scores.values()))
    
    veto_keys = ["CS-1", "CS-2", "IS-1"]
    i_veto = 0
    veto_reason = "None"
    for key in veto_keys:
        if all_scores.get(key, 0) < 0:
            i_veto = 1
            veto_reason = f"Key '{key}' with score {all_scores.get(key)}"
            break
            
    final_reward = (1 - i_veto) * r_base + i_veto * p_veto
    print(f"DEBUG: R_base={r_base}, I_veto={i_veto} (Reason: {veto_reason}), P_veto={p_veto} -> Final Reward={final_reward}")

    return {
        "overall": final_reward, "r_base": r_base, "i_veto": i_veto,
        "CS_1": all_scores.get("CS-1", 0.0),
        "CS_2": all_scores.get("CS-2", 0.0),
        "SGA_1": all_scores.get("SGA-1", 0.0),
        "SGA_2": all_scores.get("SGA-2", 0.0),
        "SGA_3": all_scores.get("SGA-3", 0.0),
        "IS_1": all_scores.get("IS-1", 0.0),
        "IS_2": all_scores.get("IS-2", 0.0),
        "IS_3": all_scores.get("IS-3", 0.0),
    }

def compute_scores_in_batch(
    batch: List[Dict[str, Any]], 
    max_workers: int = 16  # Default concurrency, can be overridden from the script
) -> List[Dict[str, Any]]:
    """
    This is the new entry point for batch processing.
    It takes a list of samples and uses a ThreadPoolExecutor to call 
    the single-sample reward function in parallel.
    """
    # The framework provides a list of dictionaries.
    # The original function expects a TypedDict, but it works with regular dicts too.
    
    print(f"🚀 Received a batch of {len(batch)} items. Processing in parallel with up to {max_workers} workers...")
    
    results = [None] * len(batch) # Pre-allocate list to store results in order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping from future to index to put results back in order
        future_to_index = {
            executor.submit(calculate_comprehensive_veto_reward, item): i 
            for i, item in enumerate(batch)
        }
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                print(f"Item at index {index} generated an exception: {exc}")
                # Handle error: you could assign a default penalty score
                # For now, we'll just let it be None and handle it later if needed
                pass

    print(f"✅ Finished processing batch of {len(batch)} items.")
    # Filter out any potential failed jobs, though ideally all should succeed
    return [res for res in results if res is not None]
