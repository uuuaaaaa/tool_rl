import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer

def format_reward(response: str) -> float:
    pattern = re.compile(
    r"^"  # 开始
    r"(?:<think>.*</think>)*"  # 匹配零次或多次 <think>.*</think>
    r"(?:<tool>.*</tool>)*"  # 匹配零次或多次 <tool>.*</tool>
    r"(?:<information>.*</information>)*"  # 匹配一次或多次 <think>.*</think>
    r"<answer>(.*?)</answer>"  # 匹配 <answer>(.*?)</answer>
    r"$",  # 结束
    re.DOTALL  # 匹配任意字符，包括换行符
)
    format_match = re.fullmatch(pattern, response)
    # match = re.search(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    # answer = extract_boxed_content(response)
    pattern = r'<(answer)>(.*?)</\1>'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        last_match = matches[-1]
        answer = last_match[1].strip()  # Extract the content inside the last tags
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    else:
        return 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    # breakpoint()
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    i=0
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "accuracy": accuracy_score,
                "format":format_score
            }
        )
        i=i+1
    print(i)
    return scores
