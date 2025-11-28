import re
from typing import Any, Dict, List
from mathruler.grader import extract_boxed_content, grade_answer
def format_reward(response: str) -> float:
    pattern = re.compile(
        # r".*<search>.*?</search>.+<answer>.*?</answer>.*",
        r"<search>.*?</search>.+<answer>.*?</answer>",
        re.DOTALL
    )
    return 1.0 if re.search(pattern, response) else 0.0
def accuracy_reward(response: str, ground_truth: str) -> float:
    pattern = r'<(answer)>(.*?)</\1>'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        answer = matches[-1][1].strip()
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    return 0.0

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    # breakpoint()
    i=0
    scores = []
    for reward_input in reward_inputs:
        try:
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
            format_score = format_reward(response)
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
            overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
            scores.append({"overall": overall_score, "accuracy": accuracy_score, "format": format_score})
            # print(overall_score)
            i=i+1
        except Exception as e:
            print(f"Error processing reward input: {e}")
            scores.append({"overall": 0.0, "accuracy": 0.0, "format": 0.0})
    print(i)
    return scores