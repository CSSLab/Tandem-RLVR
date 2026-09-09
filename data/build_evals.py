"""Rebuild the evaluation parquets under data/eval.

The seven parquets ship with the repo; this script is how they were made and how
they can be checked. Rebuilding them is not required and not recommended: the
upstream HuggingFace datasets can change under you, and row order has to match
the paper's artifacts. Compare a rebuild against the checksums in
the checksums below before using it.

One set has no public source. `olympiad` was built from a local parquet on a
source that is not publicly available, so it is left out unless --olympiad-src points at
that file.

    python data/build_evals.py                       # the six rebuildable sets
    python data/build_evals.py --sets math500 aime24 # a subset
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]

# Kept identical to the same constant in data/build_deepscaler.py. Training and
# evaluation prompts must carry the same instruction or the verifier grades a
# format the policy was never trained to emit.
BOXED_INSTR = (
    " Solve the following math problem step by step. Put your final answer "
    "inside \\boxed{}, like \\boxed{42} or \\boxed{\\frac{1}{2}}."
)

# Answers the boxed answer verifier cannot grade: a multiple choice letter, an
# interval, an expression with a free variable, a clock time.
AMC24_DROP = {"(0, \\frac{1}{2})", "[\\frac{3}{4}, \\frac{7}{8}]",
              "\\frac{\\pi}{2} - 2\\alpha", "D"}
AMC25_DROP = {"4{:}30", "k"}

EXPECT_ROWS = {
    "aime24": 30,
    "aime25": 30,
    "aime26": 30,
    "amc23_25": 121,
    "math500": 500,
    "minerva": 272,
    "olympiad": 581,
}


def make_row(idx, problem, answer, data_source, extra=None):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": problem + BOXED_INSTR}],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(answer), "style": "rule"},
        "extra_info": {"index": idx, **(extra or {})},
    }


def write(rows, name, out_root):
    if len(rows) != EXPECT_ROWS[name]:
        raise SystemExit(
            f"{name}: built {len(rows)} rows, expected {EXPECT_ROWS[name]}. The "
            f"upstream dataset has changed; this build does not match the "
            f"shipped parquet."
        )
    path = out_root / name / "test.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  {name}: n={len(rows)} -> {path}")


def first_split(ds):
    return ds[list(ds.keys())[0]]


def build_aime24(out_root, args):
    rows = []
    for r in load_dataset("math-ai/aime24", split="test"):
        m = re.search(r"\\boxed\{([^{}]+)\}", r["solution"])
        if not m:
            raise RuntimeError(f"aime24 missing boxed answer: {r}")
        rows.append(make_row(len(rows), r["problem"], m.group(1), "aime24"))
    write(rows, "aime24", out_root)


def build_aime25(out_root, args):
    rows = [make_row(i, r["problem"], r["answer"], "aime25")
            for i, r in enumerate(load_dataset("math-ai/aime25", split="test"))]
    write(rows, "aime25", out_root)


def build_aime26(out_root, args):
    rows = [make_row(i, r["problem"], r["answer"], "aime26")
            for i, r in enumerate(first_split(load_dataset("MathArena/aime_2026")))]
    write(rows, "aime26", out_root)


def build_amc23_25(out_root, args):
    rows = []
    for r in load_dataset("math-ai/amc23", split="test"):
        rows.append(make_row(len(rows), r["question"], r["answer"], "amc23_25",
                             {"subset": "amc23"}))
    for r in first_split(load_dataset("rawsh/2024_AMC12")):
        if r["answer"] not in AMC24_DROP:
            rows.append(make_row(len(rows), r["problem"], r["answer"], "amc23_25",
                                 {"subset": f"amc24_{r['exam'].split()[-1]}"}))
    for r in first_split(load_dataset("sonthenguyen/amc12-2025-non-figure")):
        if r["answer"] not in AMC25_DROP:
            rows.append(make_row(len(rows), r["question"], r["answer"], "amc23_25",
                                 {"subset": "amc25"}))
    write(rows, "amc23_25", out_root)


def build_math500(out_root, args):
    rows = [make_row(i, r["problem"], r["answer"], "math500")
            for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))]
    write(rows, "math500", out_root)


def build_minerva(out_root, args):
    rows = [make_row(i, r["question"], r["answer"], "minerva")
            for i, r in enumerate(load_dataset("math-ai/minervamath", split="test"))]
    write(rows, "minerva", out_root)


def build_olympiad(out_root, args):
    if not args.olympiad_src:
        raise SystemExit(
            "olympiad cannot be rebuilt without --olympiad-src. Its source was a "
            "local parquet with no public "
            "dataset reproduces its row order. Use the shipped copy at "
            "data/eval/olympiad/test.parquet, whose checksum is in "
            "the shipped parquet."
        )
    df = pd.read_parquet(args.olympiad_src)
    rows = [make_row(i, r["problem"], r["reward_model"]["ground_truth"], "olympiad")
            for i, r in df.iterrows()]
    write(rows, "olympiad", out_root)


# olympiad is built last so that a run without --olympiad-src still writes every
# set it can before it stops.
BUILDERS = {
    "aime24": build_aime24,
    "aime25": build_aime25,
    "aime26": build_aime26,
    "amc23_25": build_amc23_25,
    "math500": build_math500,
    "minerva": build_minerva,
    "olympiad": build_olympiad,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "eval",
                    help="directory to write <set>/test.parquet under")
    ap.add_argument("--olympiad-src", type=Path, default=None,
                    help="parquet to rebuild the olympiad set from")
    ap.add_argument("--sets", nargs="+", choices=sorted(BUILDERS),
                    default=[k for k in BUILDERS if k != "olympiad"],
                    help="which sets to build")
    args = ap.parse_args()

    for name in BUILDERS:
        if name in args.sets:
            BUILDERS[name](args.out, args)
    print("done.")


if __name__ == "__main__":
    main()
