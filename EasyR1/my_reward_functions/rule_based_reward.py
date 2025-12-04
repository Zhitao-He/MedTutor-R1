
import re
from typing import TypedDict, Optional

# --- Type definitions (no change here) ---
class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str

class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

# --- Helper Function 1: Check for all required tags ---
def check_structural_integrity(response: str) -> bool:
    """
    Checks for the presence of all 5 required structural tags using regular expressions.
    """
    patterns = [
        r"<think_history>",
        r"<think_question>",
        r"<think_student[ >]",  # Matches <think_student> or <think_student ...>
        r"<think_group>",
        r"<think_image>"
    ]
    
    for pattern in patterns:
        if not re.search(pattern, response):
            print(f"DEBUG: Structural check failed. Could not find pattern: '{pattern}'")
            return False
            
    return True

# --- Helper Function 2: Check for student name/tag consistency ---
def check_student_consistency(prompt: str, response: str) -> bool:
    """
    Checks for consistency between student names in the prompt and <think_student> tags.
    """
    try:
        analysis_section = prompt.split("Current Student Analyses")[1]
        student_names_from_prompt = re.findall(r"-\s*(\w+):", analysis_section)
    except IndexError:
        student_names_from_prompt = []

    think_student_blocks = re.findall(r"(<think_student.*?>.*?</think_student>)", response, re.DOTALL)
    
    # Check 1: Mismatch in counts
    if len(student_names_from_prompt) != len(think_student_blocks):
        print(f"DEBUG: Student consistency check failed. Reason: Mismatch in counts. "
              f"Found {len(student_names_from_prompt)} students in prompt "
              f"but {len(think_student_blocks)} <think_student> blocks in response.")
        return False
        
    # Check 2: Handle case with zero students
    if not student_names_from_prompt:
        return True # If no students in prompt and no blocks in response, it's consistent.

    # Check 3: Every student name must be mentioned
    all_blocks_text = "".join(think_student_blocks)
    missing_students = []
    for name in student_names_from_prompt:
        if not re.search(r'\b' + re.escape(name) + r'\b', all_blocks_text, re.IGNORECASE):
            missing_students.append(name)
    
    if missing_students:
        print(f"DEBUG: Student consistency check failed. Reason: The following student names "
              f"were not found in any <think_student> block: {missing_students}")
        return False

    return True

# --- Main Reward Function: Combines both checks ---
def calculate_comprehensive_reward(reward_input: RewardInput, **kwargs) -> RewardScore:
    """
    Calculates a comprehensive reward by performing two checks:
    1. Structural Integrity: Are all 5 required tags present?
    2. Student Consistency: Does the student analysis match the prompt?
    """
    prompt = reward_input["ground_truth"]
    response = reward_input["response"]
    
    # Perform Check 1: Structural Integrity
    is_structurally_correct = check_structural_integrity(response)
    
    # Perform Check 2: Student Consistency
    is_student_consistent = check_student_consistency(prompt, response)

    # Final reward is 1.0 only if BOTH checks pass
    if is_structurally_correct and is_student_consistent:
        reward = 1.0
        print(f"Comprehensive Reward Check: PASSED. Final Reward: {reward}")
    else:
        reward = 0.0
        print(f"Comprehensive Reward Check: FAILED. Final Reward: {reward}")

    return {
        "overall": reward,
        "format": reward,
        "accuracy": 0.0,
    }
