import re

_SOLUTION_CLIP_CHARS = 300

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        # Match #### followed by optional non-digit chars (currency symbols, spaces), then the number
        solutions = re.findall("#### ([^0-9\\-]*)(\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution - solutions contains tuples of (prefix, number)
            # we only want the number part (index 1), then strip commas
            final_answer = solutions[-1][1].replace(",", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == str(ground_truth):
            return score
        else:
            return format_score


def main():
    test_cases = [
        {
            "solution": "#### 300",
            "ground_truth": "300",
            "expected": 1.0,
            "description": "Basic correct format"
        },
        {
            "solution": "#### $300",
            "ground_truth": "300",
            "expected": 1.0,
            "description": "Dollar sign should be stripped"
        },
        {
            "solution": "#### 1,000",
            "ground_truth": "1000",
            "expected": 1.0,
            "description": "Comma should be stripped"
        },
        {
            "solution": "#### $1,000",
            "ground_truth": "1000",
            "expected": 1.0,
            "description": "Dollar and comma should be stripped"
        },
        {
            "solution": "Answer: 300",
            "ground_truth": "300",
            "expected": 0.0,
            "description": "No #### format = 0.0"
        },
        {
            "solution": "#### 18",
            "ground_truth": "18",
            "expected": 1.0,
            "description": "Correct match"
        },
        {
            "solution": "#### 20",
            "ground_truth": "18",
            "expected": 0.1,
            "description": "Wrong answer but has #### format = 0.1"
        },
        {
            "solution": "The total is 320 - 20 = 300\n#### $300",
            "ground_truth": "300",
            "expected": 1.0,
            "description": "Full reasoning with $ in answer"
        },
        {
            "solution": "320 - 20 = 300.\n#### 300",
            "ground_truth": "300",
            "expected": 1.0,
            "description": "Reasoning without $"
        },
        {
            "solution": "I don't know the answer.",
            "ground_truth": "300",
            "expected": 0.0,
            "description": "No format, no answer"
        },
        {
            "solution": "#### 42.5",
            "ground_truth": "42.5",
            "expected": 1.0,
            "description": "Decimal answer"
        },
        {
            "solution": "#### -5",
            "ground_truth": "-5",
            "expected": 1.0,
            "description": "Negative number"
        },
    ]

    print("="*80)
    print("GSM8K Reward Score Robustness Test")
    print("="*80)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        score = compute_score(test["solution"], test["ground_truth"], method="strict", format_score=0.1, score=1.0)
        status = "PASS" if abs(score - test["expected"]) < 1e-6 else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n[Test {i}] {status}")
        print(f"Description: {test['description']}")
        print(f"Solution: {repr(test['solution'][:80])}")
        print(f"Ground Truth: {test['ground_truth']}")
        print(f"Expected: {test['expected']}, Got: {score}")

    print("\n" + "="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80)


if __name__ == "__main__":
    main()
