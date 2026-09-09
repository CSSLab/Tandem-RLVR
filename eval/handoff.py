"""Metric 2: reasoning step handoff. The senior and the frozen junior take
turns writing one reasoning step each, and the joint answer is graded whole.

The protocol: the senior starts, each turn generates until the paragraph break
that ends a reasoning step, the two models share one response budget, both see
the same history, and a chain finishes when the budget runs out or a model
emits its end of sequence token.

PROVENANCE. Three copies of this loop exist in the source tree and they
differ, so this file is explicit about what came from where.
  * Semantics come from tandem-eval/dose/p2_hr.py: MAX_ROUNDS 150, the stop
    and completion rules in advance(), the per turn seeds, and one prompt
    string built with the senior's chat template and fed to both engines. That
  * GPU placement and the budget cap:
    one engine per card so the senior and the junior are never co-resident,
    and a per chain budget that also fits the context window, which matters on
    benchmarks with long prompts. Its OOD junior template branch is not here;
    this repo has one prompt construction route, in common.chat_prefix.
  * cleanup/scripts/v2_handoff_temp.py runs the same loop at MAX_ROUNDS 200.
"""
import argparse
import sys

import common
import config

MAX_ROUNDS = 150
DEFAULT_JUNIOR = "Qwen/Qwen3-4B-Instruct-2507"

# Per engine memory fraction. 0.85 fits one engine per card; 0.42 leaves room for
# both engines on a single card, which is the fallback when only one is free.
UTIL_PER_CARD = 0.85
UTIL_SHARED_CARD = 0.42

# turn indexes the engine list, so the senior always writes the first step.
SENIOR = 0


def make_chains(problems, prompts, budgets, n):
    chains = []
    for p, base, budget in zip(problems, prompts, budgets):
        for s in range(n):
            chains.append(
                {
                    "p": p,
                    "base": base,
                    "budget": budget,
                    "text": "",
                    "turn": SENIOR,
                    "done": False,
                    "used": 0,
                    "seed": 1000 * p["idx"] + s,
                }
            )
    return chains


def advance(chain, seg):
    chain["text"] += seg.text
    chain["used"] += len(seg.token_ids)
    if chain["used"] >= chain["budget"] or seg.finish_reason == "length":
        chain["done"] = True
    elif seg.stop_reason is None:
        # Stopped for a reason other than the paragraph break, so the model
        # ended the answer and there is nothing to hand over.
        chain["done"] = True
    else:
        chain["turn"] = 1 - chain["turn"]


def run_turn(llm, active):
    prompts = [c["base"] + c["text"] for c in active]
    params = [
        config.sampling_params(
            seed=c["seed"] + c["used"],
            max_tokens=max(1, c["budget"] - c["used"]),
            stop=["\n\n"],
            include_stop_str_in_output=True,
        )
        for c in active
    ]
    outs = llm.generate(prompts, params, use_tqdm=False)
    for c, o in zip(active, outs):
        advance(c, o.outputs[0])


def run_all_rounds(engines, chains):
    for _ in range(MAX_ROUNDS):
        moved = False
        for turn, llm in enumerate(engines):
            active = [c for c in chains if not c["done"] and c["turn"] == turn]
            if active:
                run_turn(llm, active)
                moved = True
        if not moved:
            return


def build_engines(senior, junior, util, single_gpu):
    devices = common.visible_devices()
    if single_gpu:
        return [common.build_engine(m, util) for m in (senior, junior)]
    if len(devices) < 2:
        raise SystemExit(
            "handoff needs two visible GPUs so the senior and the junior are on "
            "separate cards. Set CUDA_VISIBLE_DEVICES to two ids, or pass "
            "--single-gpu to co-locate both engines on one card at a lower "
            f"memory fraction (default {UTIL_SHARED_CARD})."
        )
    return [
        common.build_engine(m, util, device=d)
        for m, d in ((senior, devices[0]), (junior, devices[1]))
    ]


def check_vocab(senior_tok, junior_tok, allow_mismatch):
    """The loop passes raw text between the two engines and accounts the shared
    budget in tokens, so a vocabulary mismatch corrupts both quietly."""
    same = senior_tok.get_vocab() == junior_tok.get_vocab()
    print(
        f"[tok] senior vocab {len(senior_tok.get_vocab())} | "
        f"junior vocab {len(junior_tok.get_vocab())} | identical={same}"
    )
    if not same and not allow_mismatch:
        raise SystemExit(
            "senior and junior tokenizers differ; pass --allow-vocab-mismatch "
            "only if you have decided what that means for your pair."
        )
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senior", required=True)
    ap.add_argument("--junior", default=DEFAULT_JUNIOR, help="the frozen partner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="chains per problem")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--benchmarks", default=",".join(common.DEFAULT_SETS))
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--single-gpu", action="store_true", help="both engines on one card")
    ap.add_argument("--allow-vocab-mismatch", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    util = args.gpu_util
    if util is None:
        util = UTIL_SHARED_CARD if args.single_gpu else UTIL_PER_CARD

    sets = common.parse_sets(args.benchmarks)
    problems = common.load_problems(sets, args.limit)
    tok = AutoTokenizer.from_pretrained(args.senior)
    jtok = tok if args.junior == args.senior else AutoTokenizer.from_pretrained(args.junior)
    same_vocab = check_vocab(tok, jtok, args.allow_vocab_mismatch)

    prompts = [common.chat_prefix(tok, p["content"]) for p in problems]
    # The prompt string is shared, but under an allowed vocabulary mismatch the
    # two engines can disagree on its length, so budget for the longer reading.
    budgets = [
        common.response_budget(max(len(tok(pr).input_ids), len(jtok(pr).input_ids)))
        for pr in prompts
    ]

    engines = build_engines(args.senior, args.junior, util, args.single_gpu)
    chains = make_chains(problems, prompts, budgets, args.n)
    run_all_rounds(engines, chains)

    unfinished = sum(1 for c in chains if not c["done"])
    if unfinished:
        # Chains still alive here were cut by the round cap rather than by the
        # budget, which means the two models were writing very short steps.
        print(
            f"[warn] {unfinished}/{len(chains)} chains hit the MAX_ROUNDS={MAX_ROUNDS} "
            "cap before exhausting their token budget",
            file=sys.stderr,
        )

    counts, gens = [], []
    for i, p in enumerate(problems):
        sample = chains[i * args.n : (i + 1) * args.n]
        correct = [common.grade(c["text"], p["gt"]) for c in sample]
        counts.append(int(sum(correct)))
        gens.append(
            {
                "set": p["set"],
                "idx": p["idx"],
                "texts": [c["text"] for c in sample],
                "correct": correct,
            }
        )

    result = {
        "phase": "handoff",
        "senior": args.senior,
        "junior": args.junior,
        "benchmarks": list(sets),
        "limit": args.limit,
        "n": args.n,
        "sampling": config.record(),
        "max_rounds": MAX_ROUNDS,
        "unfinished_chains": unfinished,
        "same_vocab": same_vocab,
        "layout": "one engine per card" if not args.single_gpu else "both engines on one card",
        "metrics": common.metrics_by_set(problems, counts, args.n),
        "gens": gens,
    }
    common.save_json(args.out, result)
    print(f"HANDOFF {args.senior} + {args.junior}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in result["metrics"]["macro"].items()))


if __name__ == "__main__":
    main()
