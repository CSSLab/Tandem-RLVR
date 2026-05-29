#!/usr/bin/env python3
"""Download and convert MATH datasets to verl parquet under scratch/MATH/."""

from __future__ import annotations

import argparse
import os

from datasets import concatenate_datasets, load_dataset

INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_scratch() -> str:
    return os.environ.get("TANDEM_SCRATCH", os.path.join(repo_root(), "scratch"))


def make_row(data_source: str, question: str, ground_truth: str, idx: int, split: str = "test"):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": {"split": split, "index": idx},
    }


def extract_boxed_answer(text: str) -> str:
    from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

    try:
        return remove_boxed(last_boxed_only_string(text))
    except Exception:
        return str(text).strip()


def map_rows(dataset, data_source: str, question_key: str, answer_key: str, split: str = "test"):
    def process(example, idx):
        question = str(example[question_key])
        answer = example.get(answer_key, "")
        if answer_key == "solution" or (isinstance(answer, str) and "\\boxed" in answer):
            ground_truth = extract_boxed_answer(str(answer))
        else:
            ground_truth = str(answer).strip()
        return make_row(data_source, question, ground_truth, idx, split=split)

    return dataset.map(
        process,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=data_source,
    )


def save(dataset, out_dir: str, filename: str = "test.parquet"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    dataset.to_parquet(path)
    print(f"  wrote {path} ({len(dataset)} rows)")


def build_deepscaler():
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    return map_rows(ds, "deepscaler", "problem", "answer", split="train")


def build_amc_23_25():
    amc23 = load_dataset("math-ai/amc23", split="test")
    amc24 = load_dataset("rawsh/2024_AMC12", split="train")
    amc25 = load_dataset("sonthenguyen/amc12-2025-non-figure", split="train")

    parts = [
        map_rows(amc23, "amc_23_25", "question", "answer"),
        map_rows(amc24, "amc_23_25", "problem", "answer"),
        map_rows(amc25, "amc_23_25", "question", "answer"),
    ]
    merged = concatenate_datasets(parts)
    # Re-index after concat.
    return merged.map(
        lambda ex, idx: {**ex, "extra_info": {**ex["extra_info"], "index": idx}},
        with_indices=True,
        desc="reindex amc_23_25",
    )


def build_aime_24_26():
    parts = []
    for hf_name, answer_key in [
        ("math-ai/aime24", "solution"),
        ("math-ai/aime25", "answer"),
        ("math-ai/aime26", "answer"),
    ]:
        ds = load_dataset(hf_name, split="test")
        parts.append(map_rows(ds, "aime_24_26", "problem", answer_key))
    merged = concatenate_datasets(parts)
    return merged.map(
        lambda ex, idx: {**ex, "extra_info": {**ex["extra_info"], "index": idx}},
        with_indices=True,
        desc="reindex aime_24_26",
    )


def build_minerva():
    ds = load_dataset("math-ai/minervamath", split="test")
    return map_rows(ds, "minerva", "question", "answer")


def build_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return map_rows(ds, "math500", "problem", "answer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch",
        default=default_scratch(),
        help="Scratch root (default: $TANDEM_SCRATCH or <repo>/scratch)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["deepscaler", "amc_23_25", "aime_24_26", "minerva", "math500", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    math_root = os.path.join(args.scratch, "MATH")
    tasks = {
        "deepscaler": (build_deepscaler, os.path.join(math_root, "deepscaler"), "train.parquet"),
        "amc_23_25": (build_amc_23_25, os.path.join(math_root, "amc_23_25"), "test.parquet"),
        "aime_24_26": (build_aime_24_26, os.path.join(math_root, "aime_24_26"), "test.parquet"),
        "minerva": (build_minerva, os.path.join(math_root, "minerva"), "test.parquet"),
        "math500": (build_math500, os.path.join(math_root, "math500"), "test.parquet"),
    }

    selected = set(tasks) if "all" in args.only else set(args.only)
    print(f"Preparing datasets under {math_root}")
    for name in tasks:
        if name not in selected:
            continue
        builder, out_dir, filename = tasks[name]
        print(f"[{name}]")
        dataset = builder()
        save(dataset, out_dir, filename)

    print("Done.")


if __name__ == "__main__":
    main()
