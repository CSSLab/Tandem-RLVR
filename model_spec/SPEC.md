# Tandem Training — Development Specification

**Project**: `verl-llm-tandem` · Scratch-space tandem rollout prototypes + verl fork
**Spec date**: 2026-03-12
**Authority**: This file overrides all other instructions for code editing in this repo.

---

## 1. Code Style

- No `#` inline comments, no docstrings, no AI-generated boilerplate.
- Cyclomatic complexity (**CC**) <= 10 per function. Measure with:
  ```
  radon cc -s -n C <file.py>
  ```
  Any function reported at grade C or above must be refactored before merging.
- No unused imports, no dead code blocks left in.
- No type annotations added to code paths that were not already annotated.
- Prefer explicit over implicit; no magic numbers without named constants.

---

## 2. Editing Original Source Files

"Original source" means any file that was **not** authored as part of the tandem work — this includes:
- `verl/` (upstream verl library fork inside this repo)
- `verl/verl/trainer/ppo/ray_trainer.py`, `verl/verl/workers/fsdp_workers.py`, etc.
- Any installed library file under `site-packages/`

**Rule**: Never delete original lines. To disable or modify them:
```python
# [MODIFIED 2026-MM-DD <reason>]
# original_line_here()
new_line_here()
```
The comment marker format is: `# [MODIFIED YYYY-MM-DD <short reason>]`

---

## 3. Adding New Features

Workflow for every new feature or non-trivial change:

1. **Write the module** (in `scratch/tandem/` or new location as appropriate).
2. **Write a unit/smoke test** covering >= 3 edge cases (see S3.1 for what counts).
3. **Run the test** and confirm all cases pass.
4. **Write a result summary** to `model_spec/results/YYYY-MM-DD_<slug>.md` (see S3.2).
5. **Delete the test file(s)** — they live only during the verification step.

### 3.1 Test requirements

- Must be runnable without multi-GPU (CPU or single GPU only).
- Must cover: happy path, boundary/edge input, and at least one failure/error case.
- Use `assert` or `pytest` — either is fine.
- Name test files `_test_<slug>.py` (leading underscore so they stay out of imports).

### 3.2 Result file format

```markdown
# Test Results — <slug>
**Date**: YYYY-MM-DD
**Module**: scratch/tandem/<file.py>

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | happy path  | PASS   |
| 2 | edge case   | PASS   |
| 3 | error path  | PASS   |

## Notes
<any observations, caveats, known limitations>
```

All result files live in `model_spec/results/`.

---

## 4. Changelogs

Every coding session that touches `scratch/tandem/` or `verl/` must produce a changelog file at:
```
model_spec/changelog/YYYY-MM-DD_<slug>.md
```

Format:
```markdown
# Changelog — <slug>
**Date**: YYYY-MM-DD

## Changed
- <file>: <what and why>

## Added
- <file>: <what and why>

## Removed / commented out
- <file>: <original lines commented, reason>

## Open issues / TODOs
- <any known gaps>
```

---

## 5. Directory Map

```
verl-llm-tandem/
    Tandem_Training_NIPS_2026.pdf   <- paper artifact (top-level reference)
    vllm_source/                     <- patched vLLM 0.8.5 (canonical TRL backend)
        vllm/
            v1/
                worker/tandem.py            <- TandemModelManager (junior model + KV cache)
                sample/tandem_sampler.py    <- TandemSampler (5 selection strategies)
                worker/gpu_model_runner.py  <- dual-forward wiring [MODIFIED]
                worker/gpu_worker.py        <- junior TP group init [MODIFIED]
                core/sched/scheduler.py     <- authorship-mask propagation [MODIFIED]
                engine/output_processor.py  <- per-request mask accumulation [MODIFIED]
                engine/__init__.py          <- per-step mask field [MODIFIED]
                outputs.py                  <- mask in v1 SamplerOutput [MODIFIED]
            config.py                       <- TandemConfig dataclass [MODIFIED]
            outputs.py                      <- mask in top-level outputs [MODIFIED]
            engine/arg_utils.py             <- create_tandem_config factory [MODIFIED]
            distributed/parallel_state.py   <- junior TP group lifecycle [MODIFIED]
    verl/                            <- upstream verl 0.5.0 fork (original source)
        verl/
            workers/
                rollout/vllm_rollout/vllm_rollout_spmd.py  <- mask -> DataProto [MODIFIED]
                actor/dp_actor.py           <- senior-only mask in PG loss [MODIFIED]
            trainer/ppo/ray_trainer.py      <- best-ckpt + pass@N val [MODIFIED]
        run_tandem_native_grpo_deepscaler.sh  <- canonical TRL launch (paper config)
        run_tandem_native_grpo_math.sh
        run_tandem_native_grpo_gsm8k.sh
        run_vanilla_grpo_deepscaler.sh        <- baseline GRPO launches
        run_vanilla_grpo_math.sh
        run_vanilla_grpo_gsm8k_benchmark.sh
    tandem_eval/                     <- evaluation suite (handoff robustness, solo, SD, etc.)
    archive/                         <- gitignored R&D history (early prototypes)
        tandem/
            tandem_rollout.py            <- HF-loop rollout (ThreadPoolExecutor)
            tandem_rollout_optimized.py  <- CUDA-stream optimized HF-loop
            tandem_rollout_vllm.py       <- vLLM-backed step-by-step rollout
            tandem_rollout_ray.py        <- Ray actor-based junior
            tandem_rollout_shared_kv.py  <- shared KV-cache experiment
            tandem_worker.py             <- early FSDP integration
            frozen_model_actor.py        <- Ray remote junior
    scratch/                         <- gitignored: datasets, checkpoints, wandb secrets
    model_spec/
        SPEC.md          <- this file (authoritative for code editing)
        TERMINOLOGY.md   <- paper <-> code naming (authoritative for rename)
        results/         <- per-feature test result summaries
        changelog/       <- per-session change logs
```

---

## 6. Codebase Summary

The canonical Tandem Reinforcement Learning (TRL) implementation lives in two patched trees inside this repo: `vllm_source/` (dual-decoder backend, paper §A.1.1) and `verl/` (rollout extractor + senior-only mask in the actor, paper §3.4). This is what the paper trained. Earlier auto-memory references to a separate `/datadrive/difan/verl-tandem/recipe/tandem/` production repo are obsolete — that path does not exist on this host, and this repo is the single source of truth for the TRL recipe.

All identifiers use senior/junior vocabulary per `TERMINOLOGY.md`. The notes below already use the target vocabulary even where the current source still spells things `primary`/`frozen`; the sweeping rename to converge spelling lands in a separate dedicated commit.

---

### 6.1 `vllm_source/vllm/v1/worker/tandem.py` — `TandemModelManager`

Holds the frozen junior model. On engine init, builds a `VllmConfig` copy with the junior model id, resolves the junior device (`junior_gpu_devices[tp_rank]` or `tp_rank + tp_size` by default), and loads the junior with the layer-name prefix `JUNIOR_PREFIX = "tandem_junior."`. The prefixed layers are merged into the senior's `static_forward_context` so the engine's forward routing can dispatch them without colliding with senior layers. A separate junior KV-cache tensor is allocated on the junior device with shape matching the senior's paged-attention block layout. `junior_forward(...)` runs the junior under `_junior_tp_context()` (a context manager that swaps to the junior's TP group when one is configured) and returns junior logits at the same logits-indices as the senior's forward — ready for the sampler.

### 6.2 `vllm_source/vllm/v1/sample/tandem_sampler.py` — `TandemSampler`

Per step: samples once from `senior_logits` and once from `junior_logits` (both via the standard `Sampler`), computes a per-batch `use_senior` boolean via `_select(...)`, and emits `chosen_tokens = where(use_senior, senior_tokens, junior_tokens)` plus `authorship_mask = use_senior.int32`. Five selection strategies, of which `word` is the paper default:

- `word` — per-orthographic-boundary Bernoulli(p) handoff with `max_gap_tokens=32` cap (paper §3.3, §A.1.2). Uses `sampling_metadata.generators[i]` for reproducible per-request draws. State (active model, tokens since last switch, prev length) cached in `_word_state` keyed by `id(output_token_ids)`; resumes per step instead of replaying.
- `bernoulli` — per-token Bernoulli(p). Token-granularity ablation; not the paper.
- `sentence` — handoff at `\n\n`-ending boundary tokens; paragraph-granularity ablation.
- `chunk` — fixed `chunk_size` per side, deterministic cycle.
- `alternating` — strict per-step toggle.

### 6.3 `vllm_source/vllm/config.py` — `TandemConfig`

Dataclass exposed to verl via `engine_kwargs.vllm.tandem_config.*`. Fields cover junior model id, junior device(s), Bernoulli `prob_senior`, selection strategy, `chunk_size`, `max_gap_tokens`, `boundary_token_ids`, and the TP/dtype/quantization knobs the junior may diverge from the senior on. `__post_init__` validates the strategy against the literal whitelist and enforces `boundary_token_ids` presence for `sentence`/`word`. Mutually exclusive with vLLM speculative decoding (enforced in `engine/arg_utils.py`).

### 6.4 `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — rollout extractor

Converts the OmegaConf `tandem_config` to a plain dict before handing to `LLM(...)`. Auto-resolves `boundary_token_ids` from the active tokenizer when the strategy is `sentence` (token IDs whose decoded surface ends with `\n\n`) or `word` (token IDs whose decoded surface starts with the BPE leading-space marker, ~53k of 151k IDs for Qwen3). After generation, reads `output.outputs[k].authorship_mask` from each completion and pads to `response_length`; the result is added to the rollout `DataProto` as a tensor field consumed by the actor.

### 6.5 `verl/workers/actor/dp_actor.py` — senior-only mask in PG loss

Selects the authorship-mask field from the rollout batch, builds a per-position weight (1 at senior positions, `junior_token_loss_weight` at junior positions; paper sets this to 0 → senior-only loss), and elementwise-multiplies into `response_mask` before the standard policy-gradient sum. Also emits four per-microbatch metrics: `tandem/senior_token_fraction`, `tandem/junior_token_fraction`, `tandem/switches_per_seq`, `tandem/tokens_per_sent`. With `junior_token_loss_weight=0`, the loss is formally identical to vanilla GRPO restricted to senior positions, matching eq. (1) of the paper.

### 6.6 `verl/run_tandem_native_grpo_deepscaler.sh` — canonical TRL launch

Paper-aligned config: Qwen3-4B-Instruct-2507 as both senior init and self-paired junior, `selection_strategy=word`, `prob_senior=0.5`, `max_gap_tokens=32`, `junior_token_loss_weight=0`, GRPO with `kl_loss=False`, `entropy_coeff=0`, temp=0.6, 2× A100 80GB with `frozen_gpu_devices=[1]`. Sources wandb credentials from `scratch/wandb_secrets.env` (gitignored). The other `run_tandem_native_grpo_{math,gsm8k}.sh` variants are off-paper ablations preserved for reference.

---

## 7. How the senior-only mask reaches the loss

A trace through the data flow on a single training step:

1. `TandemSampler.forward(senior_logits, junior_logits, sampling_metadata)` returns the chosen-token tensor and a per-batch `use_senior` bool. The bool is reshaped to `[batch, 1]` int32 → the per-step authorship.
2. `gpu_model_runner` stores it on `ModelRunnerOutput.authorship_mask` alongside the sampled tokens.
3. `scheduler.update_from_output(...)` distributes the per-batch mask to each `EngineCoreOutput` as `new_authorship_mask`.
4. `output_processor` accumulates per-request `new_authorship_mask` deltas into a list, surfacing the full per-token mask in the completion's `authorship_mask` field.
5. `vllm_rollout_spmd._generate(...)` reads `output.outputs[k].authorship_mask` per sample, pads to `response_length`, and adds it to the rollout `DataProto`.
6. `dp_actor._forward_micro_batch_with_log` looks it up, weights `response_mask` by it (1 at senior, `junior_token_loss_weight` at junior), and uses the weighted mask in the standard PG sum.

Step 6 is the only place GRPO touches TRL; everything upstream is plumbing for that one elementwise multiply. The rest of `ray_trainer.fit()` is unmodified — TRL is a pure rollout-structure change, not a loss change, exactly as paper §3.4 claims.

`ray_trainer.py` carries two unrelated edits — best-checkpoint tracking for HF-format export and per-dataset pass@N val-core reporting — neither tandem-specific.

---

## 8. Implementation paths

Three paths exist in the repo, with different status. Only the first is what the paper trained.

| Path | Location | Status |
|---|---|---|
| **vLLM-native dual-decoder** (canonical TRL) | `vllm_source/vllm/v1/*` + `verl/verl/workers/{rollout,actor}/*` | **What the paper trained.** Active. Used by every tracked `run_*.sh`. ~2× single-model latency (paper §5.1). |
| Early HF-loop prototypes | `archive/tandem/*.py` (gitignored) | Superseded. ~30× single-model latency at short contexts; OOM at 512+ tokens on 80 GB GPUs (paper §A.1). Kept on disk for R&D reference, not for active development. |
| Off-paper selection-strategy ablations | `tandem_sampler.py` strategies `bernoulli`, `chunk`, `alternating`, `sentence` | Selectable at runtime via `selection_strategy=…`. Not used by the paper (which uses `word`). Retained as a contribution surface — they expose the rollout-structure design axis the paper argues is under-explored. |

Canonical TRL configuration: `selection_strategy=word`, `prob_senior=0.5`, `max_gap_tokens=32`, `junior_token_loss_weight=0`, self-paired junior. Any deviation belongs to an explicit ablation script, not the default.
