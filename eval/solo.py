"""Metric 1: solo capability. The senior answers alone, pass@{1,2,4,8}.

This phase also stores its generations, which legibility.py reads.
chains of thought under a second decoding path.
"""
import argparse

import common
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="senior checkpoint, hub id or local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="samples per problem")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--benchmarks", default=",".join(common.DEFAULT_SETS))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    sets = common.parse_sets(args.benchmarks)
    problems = common.load_problems(sets, args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = common.build_engine(args.model, args.gpu_util)

    prompts = [common.chat_prefix(tok, p["content"]) for p in problems]
    outs = llm.generate(prompts, config.sampling_params(n=args.n, seed=17))

    counts, gens = [], []
    for p, o in zip(problems, outs):
        texts = [c.text for c in o.outputs]
        correct = [common.grade(t, p["gt"]) for t in texts]
        counts.append(int(sum(correct)))
        gens.append({"set": p["set"], "idx": p["idx"], "texts": texts, "correct": correct})

    result = {
        "phase": "solo",
        "model": args.model,
        "benchmarks": list(sets),
        "limit": args.limit,
        "n": args.n,
        "sampling": config.record(),
        "metrics": common.metrics_by_set(problems, counts, args.n),
        "gens": gens,
    }
    common.save_json(args.out, result)
    print(f"SOLO {args.model}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in result["metrics"]["macro"].items()))


if __name__ == "__main__":
    main()
