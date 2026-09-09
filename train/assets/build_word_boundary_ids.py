#!/usr/bin/env python3
"""Regenerate qwen3_word_boundary_ids.json, the word handoff's boundary set.

In the tandem rollout the active model is redrawn after any token in this list
(see train/tandem_grpo.sh). For Qwen3's byte-level BPE, a token starts a new
orthographic word exactly when its surface form begins with the leading-space
marker U+0120, "Ġ". There are 53,021 such ids in the Qwen3 vocabulary, far
past what one environment variable can carry, which is why the vLLM fork reads
them from a file (vllm/config/tandem.py, boundary_token_ids_path).

Tokenizer files only: no weights, no GPU, a few MB of download.

    python train/assets/build_word_boundary_ids.py            # check the shipped file
    python train/assets/build_word_boundary_ids.py --write    # rewrite it

Checked against the Qwen3-4B-Instruct-2507 tokenizer: the rule yields exactly the
shipped set of 53,021 ids, as an identical set and not merely the same count.
straight out of tokenizer.json; on that input it reproduces the shipped file
byte for byte. Nobody has yet run it with transformers installed and the model
id resolved from the hub.
"""

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUT = Path(__file__).resolve().parent / "qwen3_word_boundary_ids.json"

# Byte-level BPE renders a leading space as U+0120 rather than as a space.
LEADING_SPACE_MARKER = "Ġ"

# The set the paper's runs used. A different count means a different tokenizer,
# and word handoff is then not the schedule the results were produced under.
EXPECTED_COUNT = 53021


def build(model: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return sorted(
        token_id
        for surface, token_id in tokenizer.get_vocab().items()
        if surface.startswith(LEADING_SPACE_MARKER)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true", help="overwrite --out instead of checking it")
    args = parser.parse_args()

    ids = build(args.model)
    if len(ids) != EXPECTED_COUNT:
        raise SystemExit(f"expected {EXPECTED_COUNT} boundary ids from {args.model}, got {len(ids)}")

    # Default separators and no trailing newline: this is the exact form of the
    # shipped file, so the comparison below is a byte comparison.
    payload = json.dumps(ids)

    if args.write:
        args.out.write_text(payload)
        print(f"wrote {len(ids)} ids to {args.out}")
        return

    if not args.out.exists():
        raise SystemExit(f"{args.out} does not exist; rerun with --write")
    if args.out.read_text() != payload:
        raise SystemExit(f"{args.out} differs from what {args.model} produces")
    print(f"{args.out} matches {args.model}: {len(ids)} ids")


if __name__ == "__main__":
    main()
