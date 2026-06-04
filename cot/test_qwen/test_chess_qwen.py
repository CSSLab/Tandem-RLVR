import argparse
import json
import re
import sys
from pathlib import Path

import requests
from datasets import load_dataset

DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DATASET_NAME = "OutFlankShu/MATE_DATASET"
DEFAULT_MAX_TOKENS = 32
DEFAULT_MAX_TOKENS_THINKING = 3000
THINKING_END_TAG = "</think>"

MOVE_ANSWER_RE = re.compile(
    r"\bMove([AB])\s*:\s*([a-h][1-8][a-h][1-8][qrbn]?)\b",
    re.IGNORECASE,
)


def split_thinking_and_answer(
    content: str,
    *,
    enable_thinking: bool,
    reasoning: str | None = None,
) -> tuple[str, str, bool]:
    """Return (thinking, answer, truncated)."""
    if reasoning is not None:
        return reasoning.strip(), content.strip(), False

    if not enable_thinking:
        return "", content.strip(), False

    if THINKING_END_TAG in content:
        thinking, answer = content.rsplit(THINKING_END_TAG, maxsplit=1)
        return thinking.strip(), answer.strip(), False

    return content.strip(), "", True


def parse_move_answer(text: str) -> str | None:
    """Extract 'MoveA:uci' or 'MoveB:uci' from model output."""
    if not text:
        return None
    matches = list(MOVE_ANSWER_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    label = match.group(1).upper()
    uci = match.group(2).lower()
    return f"Move{label}:{uci}"


def normalize_expected(text: str) -> str:
    match = MOVE_ANSWER_RE.search(text)
    if not match:
        return text.strip()
    return f"Move{match.group(1).upper()}:{match.group(2).lower()}"


def build_prompt(instruction: str, user_input: str) -> str:
    return (
        f"{instruction.strip()}\n\n"
        f"{user_input.strip()}\n\n"
        "Reply with exactly one line in the format MoveA:<uci> or MoveB:<uci>."
    )


def query_qwen(
    prompt: str,
    *,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    response = requests.post(api_url, json=payload, timeout=300)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    message = choice["message"]
    return {
        "content": (message.get("content") or "").strip(),
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
    }


def iter_mate_samples(split: str, start: int, limit: int | None):
    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    for index, row in enumerate(dataset):
        if index < start:
            continue
        yield index, row
        if limit is not None and index >= start + limit - 1:
            break


def load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            indices.add(json.loads(line)["index"])
    return indices


def resolve_resume_start(args: argparse.Namespace) -> int:
    if not args.resume:
        return args.start

    if not args.save_thinking:
        raise SystemExit("--resume requires --save-thinking PATH")

    completed = load_completed_indices(Path(args.save_thinking))
    if not completed:
        print(
            f"No completed samples found in {args.save_thinking}; starting at --start {args.start}.",
            file=sys.stderr,
        )
        return args.start

    resume_start = max(completed) + 1
    print(
        f"Resuming from index {resume_start} "
        f"({len(completed)} samples already in {args.save_thinking}).",
        file=sys.stderr,
    )
    return resume_start


def evaluate(args: argparse.Namespace) -> int:
    correct = 0
    total = 0
    truncated = 0
    unparseable = 0
    thinking_file = None

    if args.save_thinking:
        thinking_path = Path(args.save_thinking)
        thinking_path.parent.mkdir(parents=True, exist_ok=True)
        thinking_file = thinking_path.open("a", encoding="utf-8")
        if not args.enable_thinking:
            print(
                "Warning: --save-thinking without --enable-thinking; "
                "thinking fields will be empty.",
                file=sys.stderr,
            )

    try:
        for index, row in iter_mate_samples(args.split, args.start, args.num_samples):
            instruction = row["instruction"]
            user_input = row["input"]
            expected = normalize_expected(row["output"])
            prompt = build_prompt(instruction, user_input)

            try:
                result = query_qwen(
                    prompt,
                    api_url=args.api_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                )
            except requests.RequestException as exc:
                print(f"[{index}] request failed: {exc}", file=sys.stderr)
                break

            thinking_text, answer_text, thinking_truncated = split_thinking_and_answer(
                result["content"],
                enable_thinking=args.enable_thinking,
                reasoning=result["reasoning"],
            )
            parsed = parse_move_answer(answer_text)
            is_correct = parsed == expected
            correct += int(is_correct)
            total += 1
            truncated += int(thinking_truncated)
            unparseable += int(parsed is None)

            if is_correct:
                status = "OK"
            elif thinking_truncated:
                status = "TRUNCATED"
            elif parsed is None:
                status = "UNPARSEABLE"
            else:
                status = "MISS"

            if thinking_file is not None:
                record = {
                    "index": index,
                    "status": status,
                    "expected": expected,
                    "answer": answer_text,
                    "parsed": parsed,
                    "correct": is_correct,
                    "thinking": thinking_text,
                    "thinking_truncated": thinking_truncated,
                    "finish_reason": result["finish_reason"],
                    "instruction": instruction,
                    "input": user_input,
                    "raw_content": result["content"],
                }
                thinking_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                thinking_file.flush()

            print(f"[{index}] {status}")
            print(f"  expected: {expected}")
            print(f"  answer:   {answer_text or '(empty)'}")
            if result["finish_reason"] == "length":
                print("  note:     generation hit max_tokens")
            if parsed and parsed != answer_text:
                print(f"  parsed:   {parsed}")
            if args.verbose:
                print(f"  raw:      {result['content'][:500]}")
            print()
    finally:
        if thinking_file is not None:
            thinking_file.close()
            print(f"Saved thinking traces to {args.save_thinking}")

    if total == 0:
        print("No samples evaluated.")
        return 1

    accuracy = 100.0 * correct / total
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    if args.enable_thinking:
        print(f"Truncated thinking: {truncated}/{total}")
    print(f"Unparseable answers: {unparseable}/{total}")
    return 0 if correct == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Qwen server on OutFlankShu/MATE_DATASET."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N rows.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of dataset rows to evaluate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Completion token budget. Defaults to 32 without thinking and "
            f"{DEFAULT_MAX_TOKENS_THINKING} with --enable-thinking."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode in chat_template_kwargs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the raw model content for debugging.",
    )
    parser.add_argument(
        "--save-thinking",
        metavar="PATH",
        help=(
            "Append one JSON object per sample to PATH for chain-of-thought "
            "analysis. Each line includes thinking, answer, prompt, and labels."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from the next index after the last saved row in "
            "--save-thinking. Requires --save-thinking."
        ),
    )
    args = parser.parse_args()
    args.start = resolve_resume_start(args)
    if args.max_tokens is None:
        args.max_tokens = (
            DEFAULT_MAX_TOKENS_THINKING
            if args.enable_thinking
            else DEFAULT_MAX_TOKENS
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
