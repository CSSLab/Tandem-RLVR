"""Pieces shared by every metric: benchmark loading, prompt construction,
grading, the pass@k estimator, engine construction and result IO.

Sampling settings are not here, they are in config.py.
"""
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"

sys.path.insert(0, str(ROOT / "reward"))
from math_boxed_reward import compute_score  # noqa: E402

# The four sets the reported dose curve was run on. All seven shipped
# benchmarks are selectable with --benchmarks.
DEFAULT_SETS = ("aime24", "aime25", "aime26", "amc23_25")
ALL_SETS = ("aime24", "aime25", "aime26", "amc23_25", "math500", "minerva", "olympiad")

KS = (1, 2, 4, 8)

# Engine context window. Prompt plus response must fit inside it, so it caps
# the config.MAX_TOKENS response budget on long prompts (see response_budget).
MAX_MODEL_LEN = 4096
CONTEXT_RESERVE = 8


# ---------------------------------------------------------------- benchmarks

def parse_sets(spec):
    """Turn a comma separated --benchmarks value into a validated tuple."""
    names = tuple(s.strip() for s in spec.split(",") if s.strip())
    unknown = [n for n in names if not (DATA / n / "test.parquet").exists()]
    if unknown:
        raise SystemExit(
            f"unknown benchmark(s) {', '.join(unknown)}; available: {', '.join(ALL_SETS)}"
        )
    return names


def load_problems(sets=DEFAULT_SETS, limit=0):
    """Load the selected benchmarks in order into one flat problem list.

    'idx' is the position in that flat list and is the join key between phases,
    so solo.py records the benchmark list and limit it ran with, and the phases
    that consume solo output rebuild the list from that record rather than from
    their own flags. --limit truncates the concatenated list, matching the
    behaviour of the scripts this suite comes from.
    """
    rows = []
    for name in sets:
        df = pd.read_parquet(DATA / name / "test.parquet")
        for pos, (_, r) in enumerate(df.iterrows()):
            rows.append(
                {
                    "set": name,
                    "row": pos,
                    "idx": len(rows),
                    "content": r["prompt"][0]["content"],
                    "gt": str(r["reward_model"]["ground_truth"]),
                }
            )
    return rows[:limit] if limit else rows


def problems_from_result(result):
    """Rebuild the exact problem list a stored result was produced over."""
    return load_problems(tuple(result["benchmarks"]), result.get("limit", 0))


# ------------------------------------------------------------------- prompts

def chat_prefix(tokenizer, content):
    """The one prompt construction route in this repo.

    enable_thinking=False is applied on the single path every metric uses. It is inert
    for Qwen3-4B-Instruct-2507, which has no thinking mode, and it keeps the
    hybrid Qwen3 checkpoints out of thinking mode if one is ever used as a
    partner. Templates that reject the argument have no thinking mode to
    disable, so the fallback is the same prompt rather than a second protocol.
    """
    messages = [{"role": "user", "content": content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **kwargs)


def response_budget(prompt_len):
    """Response tokens available to a request whose prompt is prompt_len long."""
    return min(config.MAX_TOKENS, MAX_MODEL_LEN - prompt_len - CONTEXT_RESERVE)


# ------------------------------------------------------------------- grading

def grade(text, gt):
    """1.0 if the boxed answer matches the ground truth, else 0.0."""
    return float(compute_score("math", text, gt)["acc"])


def pass_at_k(n, c, k):
    """Unbiased pass@k for c correct out of n samples, exact combinatorics."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def agg_pass_at_k(counts, n, ks=KS):
    return {
        f"pass@{k}": sum(pass_at_k(n, c, k) for c in counts) / max(len(counts), 1)
        for k in ks
        if k <= n
    }


def metrics_by_set(problems, counts, n, ks=KS):
    """Per benchmark pass@k plus the macro average over benchmarks."""
    grouped = {}
    for p, c in zip(problems, counts):
        grouped.setdefault(p["set"], []).append(c)
    by_set = {name: agg_pass_at_k(cs, n, ks) for name, cs in grouped.items()}
    out = dict(by_set)
    out["macro"] = {
        key: sum(v[key] for v in by_set.values()) / len(by_set)
        for key in next(iter(by_set.values()))
    }
    return out


# ------------------------------------------------------------------- engines

def build_engine(model, gpu_util, device=None, max_num_batched_tokens=None):
    """Construct one vLLM engine, optionally pinned to a single card.

    vLLM spawns its EngineCore as a subprocess that inherits the environment at
    construction time, so setting CUDA_VISIBLE_DEVICES around the constructor
    is what keeps two engines on separate cards.
    """
    from vllm import LLM

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        kwargs = dict(
            model=model,
            gpu_memory_utilization=gpu_util,
            enforce_eager=True,
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=True,
        )
        if max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        return LLM(**kwargs)
    finally:
        if device is not None:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved


def visible_devices():
    """The card ids this process may use, in order."""
    spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not spec:
        return []
    return [d for d in spec.split(",") if d.strip()]


# ------------------------------------------------------------------------ io

def save_json(path, obj):
    """Write via a .partial file so an interrupted run leaves no half result
    for the resume logic in run_all.sh to mistake for a finished one."""
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)
