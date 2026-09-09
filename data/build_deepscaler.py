"""Rebuild the DeepScaleR split used by every training arm.

Writes two disjoint parquets: train.parquet, and heldout.parquet, which the
training launchers pass to the trainer as its validation set.

The split has to be made here rather than by pointing validation at a benchmark,
because a checkpoint chosen on a benchmark that is later reported is chosen on
its own test set. Holding out a slice of the training distribution keeps every
reported benchmark untouched by model selection. The rows are removed from
train.parquet, so the validation slice is genuinely unseen.

Not shipped, unlike data/eval: it is 6.0 MB and rebuilds deterministically from
one HuggingFace dataset.

    python data/build_deepscaler.py
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
SRC = "agentica-org/DeepScaleR-Preview-Dataset"
EXPECT_ROWS = 40309

# Appended to every prompt, training and eval alike, so that the boxed-answer
# verifier in reward/ has something to match against.
BOXED_INSTR = (
    " Solve the following math problem step by step. Put your final answer "
    "inside \\boxed{}, like \\boxed{42} or \\boxed{\\frac{1}{2}}."
)


def make_row(idx, problem, answer):
    return {
        "data_source": "deepscaler",
        "prompt": [{"role": "user", "content": problem + BOXED_INSTR}],
        "ability": "math",
        "reward_model": {"ground_truth": str(answer), "style": "rule"},
        "extra_info": {"index": idx},
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "data" / "deepscaler",
                    help="directory to write train.parquet and heldout.parquet into")
    ap.add_argument("--heldout-n", type=int, default=1000,
                    help="rows held out of training for checkpoint selection")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the held-out draw; changing it changes the split")
    ap.add_argument("--expect-rows", type=int, default=EXPECT_ROWS,
                    help="row count to assert; 0 disables the check")
    args = ap.parse_args()

    df = load_dataset(SRC, split="train").to_pandas()
    print("total:", len(df), "cols:", list(df.columns))

    # An empty answer has no ground truth for the verifier to grade, so the row
    # would contribute a constant zero reward to its GRPO group.
    keep = df[df["answer"].astype(str).str.len() > 0].reset_index(drop=True)
    print("dropped empty-answer rows:", len(df) - len(keep))

    out = pd.DataFrame([make_row(i, r["problem"], r["answer"])
                        for i, r in keep.iterrows()])

    if args.expect_rows and len(out) != args.expect_rows:
        raise SystemExit(
            f"row count {len(out)} != expected {args.expect_rows}. The upstream "
            f"dataset has changed."
        )

    held = out.sample(n=args.heldout_n, random_state=args.seed)
    train = out.drop(held.index)
    assert not set(held.index) & set(train.index)
    held = held.reset_index(drop=True)
    train = train.reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out_dir / "train.parquet", index=False)
    held.to_parquet(args.out_dir / "heldout.parquet", index=False)
    print(f"wrote {len(train)} train + {len(held)} held out -> {args.out_dir}")
    print("sample prompt:", train.iloc[0]["prompt"][0]["content"][:200])
    print("sample ground truth:", train.iloc[0]["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()
