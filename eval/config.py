"""The single sampling configuration for every evaluation in this repo.

vLLM defaults top_k to -1, which disables the cutoff. A call site that builds
its own SamplingParams therefore drops top_k without raising anything, and the
run stays plausible while no longer matching the rest of the suite. Every eval
script builds its parameters through the factories below so that the settings
have exactly one definition.

The engine limit that bounds these budgets, MAX_MODEL_LEN, lives in
common.py next to the code that computes response budgets from it.
"""

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_TOKENS = 3000

# Refused as per call overrides: changing any of these changes the protocol,
# and a run that changes it silently is not comparable to any other run here.
DECODING_KEYS = ("temperature", "top_p", "top_k")


def sampling_params(**overrides):
    """SamplingParams for generation.

    Accepted overrides are per request bookkeeping and budget: n, seed,
    max_tokens, stop, include_stop_str_in_output. To evaluate under different
    decoding, edit the constants above once rather than at a call site.
    """
    refused = [k for k in DECODING_KEYS if k in overrides]
    if refused:
        raise ValueError(
            f"{', '.join(refused)} cannot be set per call. This repo evaluates at "
            f"temperature {TEMPERATURE}, top_p {TOP_P}, top_k {TOP_K}; change the "
            "constants in eval/config.py if you mean to change the protocol."
        )
    from vllm import SamplingParams

    kwargs = dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS)
    kwargs.update(overrides)
    return SamplingParams(**kwargs)


def scoring_params():
    """SamplingParams for teacher forced scoring, used by legibility.py.

    prompt_logprobs=0 returns the logprob of each prompt token as it actually
    appears, which is what a cross-entropy over someone else's text needs.
    max_tokens=1 is the smallest generation vLLM accepts and its output is
    discarded; the decoding knobs are pinned to inert values so that nothing
    here depends on the generation settings above.
    """
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1, prompt_logprobs=0
    )


def record():
    """The generation settings as plain data, stored in every result JSON."""
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
    }
