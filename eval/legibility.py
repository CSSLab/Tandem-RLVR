"""Metric 3: legibility to the junior. The frozen junior reads the senior's
solo chains of thought and reports its mean cross-entropy per token over them,
conditioned on the same chat prefix the senior saw.

Nothing is sampled here. Each chain of thought is fed to the junior as prompt
and scored with prompt_logprobs, so the number is a teacher forced likelihood
and does not depend on the generation settings.

The cross-entropy core
this file keeps unchanged. The batching limits come from its sibling
prompt_logprobs materialises a full vocabulary
distribution per prompt token, so the prefill batch has to stay small or a 48G
card runs out of memory.
"""
import argparse

import numpy as np

import common
import config

DEFAULT_JUNIOR = "Qwen/Qwen3-4B-Instruct-2507"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", required=True, help="the JSON solo.py wrote")
    ap.add_argument("--junior", default=DEFAULT_JUNIOR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=0, help="CoTs per problem, 0 = all")
    ap.add_argument("--gpu-util", type=float, default=0.60)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=2000, help="prompts per generate call")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    solo = common.load_json(args.solo)
    problems = common.problems_from_result(solo)
    by_idx = {g["idx"]: g for g in solo["gens"]}

    tok = AutoTokenizer.from_pretrained(args.junior)

    todo, dropped = [], 0
    for p in problems:
        prefix = common.chat_prefix(tok, p["content"])
        prefix_len = len(tok(prefix).input_ids)
        texts = by_idx[p["idx"]]["texts"]
        if args.samples:
            texts = texts[: args.samples]
        for sample_idx, cot in enumerate(texts):
            full = prefix + cot
            if len(tok(full).input_ids) >= common.MAX_MODEL_LEN - common.CONTEXT_RESERVE:
                dropped += 1
                continue
            todo.append((p, sample_idx, full, prefix_len))
    print(f"[plan] scoring {len(todo)} CoTs, dropped {dropped} that exceed the context")

    llm = common.build_engine(
        args.junior, args.gpu_util, max_num_batched_tokens=args.max_batched_tokens
    )
    params = config.scoring_params()

    per_sample = []
    for start in range(0, len(todo), args.chunk):
        batch = todo[start : start + args.chunk]
        outs = llm.generate([t[2] for t in batch], params)
        for (p, sample_idx, _, prefix_len), o in zip(batch, outs):
            # prompt_logprobs holds one entry per prompt token, the first being
            # None because nothing conditions it. Dropping the prefix leaves the
            # senior's tokens, each scored as it actually appears.
            nll = [
                -list(d.values())[0].logprob
                for d in o.prompt_logprobs[prefix_len:]
                if d
            ]
            if nll:
                per_sample.append(
                    {
                        "set": p["set"],
                        "idx": p["idx"],
                        "sample_idx": sample_idx,
                        "ce": float(np.mean(nll)),
                        "n_tokens": len(nll),
                    }
                )
        print(f"[progress] {min(start + args.chunk, len(todo))}/{len(todo)} CoTs scored", flush=True)

    by_set = {}
    for r in per_sample:
        by_set.setdefault(r["set"], []).append(r["ce"])
    ces = [r["ce"] for r in per_sample]

    result = {
        "phase": "legibility",
        "senior": solo["model"],
        "junior": args.junior,
        "solo_src": args.solo,
        "benchmarks": solo["benchmarks"],
        "limit": solo.get("limit", 0),
        "n_scored": len(ces),
        "n_dropped": dropped,
        "by_set": {name: float(np.mean(v)) for name, v in by_set.items()},
        "mean_ce": float(np.mean(ces)) if ces else None,
        "macro_ce": (
            float(np.mean([np.mean(v) for v in by_set.values()])) if by_set else None
        ),
        "per_sample": per_sample,
    }
    common.save_json(args.out, result)
    print(f"LEGIBILITY {solo['model']} read by {args.junior}")
    if result["macro_ce"] is None:
        raise SystemExit("no CoT was scored; every one exceeded the context window")
    print(f"  macro mean CE = {result['macro_ce']:.4f} nats over {len(ces)} CoTs")


if __name__ == "__main__":
    main()
