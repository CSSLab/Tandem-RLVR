import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hendrycks_math_grader import boxed_reward_fn


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        info, score = boxed_reward_fn(solution_str, str(ground_truth), fast=False)
        present = 1.0 if info.get("formatted", False) else 0.0
        correct = 1.0 if score > 0.0 else 0.0
    except Exception:
        present, correct = 0.0, 0.0
    return {"score": correct, "acc": correct, "answer_present": present}
