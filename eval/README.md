# Evaluation

Three metrics. Each is one script, one process, one JSON.

| script | measures |
|---|---|
| `solo.py` | the senior alone: pass@k over n independent samples per problem |
| `handoff.py` | senior and junior alternating at every `\n\n`, one reasoning step each, under a shared token budget, with the team's finished answer graded |
| `legibility.py` | the frozen junior's mean per-token cross-entropy, in nats, over the senior's chain of thought |

`handoff.py` needs two GPUs, one engine per card. `solo.py` and `legibility.py` need one.

## Run

```bash
MODEL=/path/to/senior TAG=step_120 RESULTS_ROOT=results/tandem bash run_all.sh
```

Runs all three in order and writes `results/tandem/step_120/{solo,handoff,legibility}.json`,
each with a `metrics` object holding that phase's numbers. A
phase whose JSON exists is skipped, so the command is safe to rerun. `JUNIOR` defaults to
`BASE_MODEL` from `train/env.sh`; `BENCHMARKS`, `N`, `LIMIT` and `GPU_UTIL` are also overridable.

Individually:

```bash
python solo.py --model $SENIOR --out results/tandem/step_120/solo.json --n 8

python handoff.py --senior $SENIOR --junior Qwen/Qwen3-4B-Instruct-2507 \
  --out results/tandem/step_120/handoff.json --n 8

python legibility.py --senior-file results/tandem/step_120/solo.json \
  --junior Qwen/Qwen3-4B-Instruct-2507 --bench aime24 \
  --out results/tandem/step_120/legibility.json
```

`legibility.py` reads the generations `solo.py` saved, so run solo first.

## Sampling

Temperature 0.7, top_p 0.8, top_k 20, max_tokens 3000, defined once in `config.py`. Every script
builds `SamplingParams` through `config.sampling_params()`, which refuses per call overrides of
the three decoding parameters. `legibility.py` does not sample; it uses `config.scoring_params()`,
temperature 0 with `prompt_logprobs`.

If you add a script, use the factory. vLLM defaults `top_k` to -1, so a script that imports the
constants but builds its own `SamplingParams` drops the top_k setting without an error.

## Benchmarks

`data/eval/` ships seven parquets. `--benchmarks` takes a comma separated list of directory names;
the default is set in `common.py`.
