import re
from typing import Any, Dict, List
from mathruler.grader import extract_boxed_content, grade_answer
import re

def format_reward(response: str) -> float:
    pattern = re.compile(
        r"^"
        # r"(?:<think>.*?</think>)*"  # 0或多个额外的think块
        # r"(?:<think>.*?</think>)?"  # 0或1个think块
        r"(?:<think>.*?</think>)+"  # 1或多个额外的think块
        r"(?:<search>.*?</search>)?"  # 0或1个search块
        # r"<think>.*?</think>"
        # r"(?:<think>.*?</think>)+"  # 1或多个额外的think块
        r"<answer>.*?</answer>"  # 必须以answer块结尾
        r"$",
        re.DOTALL
    )
    return 1.0 if re.fullmatch(pattern, response) else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    pattern = r'<(answer)>(.*?)</\1>'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        answer = matches[-1][1].strip()
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    return 0.0

# def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")
#     # breakpoint()
#     i=0
#     scores = []
#     for reward_input in reward_inputs:
#         try:
#             response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
#             format_score = format_reward(response)
#             accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
#             overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
#             scores.append({"overall": overall_score, "accuracy": accuracy_score, "format": format_score})
#             # print(overall_score)
#             i=i+1
#         except Exception as e:
#             print(f"Error processing reward input: {e}")
#             scores.append({"overall": 0.0, "accuracy": 0.0, "format": 0.0})
        
#     print(i)
#     return scores
import re
from typing import List, Dict, Any

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    
    scores = []
    print_interval = max(1, len(reward_inputs) // 15)  # 打印大约3个样本
    
    for i, reward_input in enumerate(reward_inputs):
        try:
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
            format_score = format_reward(response)
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
            overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
            scores.append({"overall": overall_score, "accuracy": accuracy_score, "format": format_score})
            
            # 每隔一定间隔打印样本
            if i % print_interval == 0:
                print(f"\n=== 样本 {i+1}/{len(reward_inputs)} ===")
                print(f"ground_truth: {reward_input['ground_truth']}")
                print(f"response: {response}")
                print(f"accuracy_score: {accuracy_score},format_score:{format_score}")
                
        except Exception as e:
            print(f"Error processing reward input at index {i}: {e}")
            scores.append({"overall": 0.0, "accuracy": 0.0, "format": 0.0})
    
    print(f"\n总共处理了 {len(scores)} 个样本")
    return scores