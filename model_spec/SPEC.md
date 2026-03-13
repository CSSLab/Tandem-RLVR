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
    verl/                        <- upstream verl fork (original source)
        verl/
            trainer/ppo/
                ray_trainer.py   <- RayPPOTrainer (core training loop)
            workers/
                fsdp_workers.py  <- ActorRolloutRefWorker base class
    scratch/
        tandem/                  <- tandem rollout prototypes (our code)
            tandem_rollout.py            <- HF-loop rollout (ThreadPoolExecutor)
            tandem_rollout_optimized.py  <- CUDA-stream optimized HF-loop
            tandem_rollout_vllm.py       <- vLLM-backed step-by-step rollout
            tandem_rollout_ray.py        <- Ray actor-based frozen model
            tandem_rollout_shared_kv.py  <- shared KV-cache experiment
            tandem_worker.py             <- TandemActorRolloutWorker (FSDP)
            frozen_model_actor.py        <- FrozenModelActor (Ray remote)
    model_spec/
        SPEC.md              <- this file (authoritative)
        results/             <- per-feature test result summaries
        changelog/           <- per-session change logs
```

---

## 6. Codebase Summary

### Relationship to `verl-tandem`

This repo (`verl-llm-tandem`) is a **parallel scratch workspace** for the tandem training project.
The production-ready tandem recipe lives in `/datadrive/difan/verl-tandem/recipe/tandem/`.
This repo contains:
- A forked copy of the verl library (`verl/`)
- Scratch prototypes under `scratch/tandem/` exploring different rollout strategies

The prototypes here are less mature than the production recipe — they use `model_a`/`model_b`
naming (vs `primary`/`tandem`), lack `TokenSelectionStrategy` integration (using raw `prob_a`
Bernoulli instead), and do not output `primary_log_probs`/`tandem_log_probs` (except the
optimized variant).

---

### `tandem_rollout.py` — `TandemRollout`

HF-style autoregressive loop using `ThreadPoolExecutor` for parallel forward passes.
Both models maintain separate KV caches. Per step: forward both models, sample from each,
Bernoulli-select which candidate token to keep. Both models are fed the same chosen token.

Output keys: `prompts`, `responses`, `input_ids`, `attention_mask`, `position_ids`, `model_a_mask`.

---

### `tandem_rollout_optimized.py` — `TandemRolloutOptimized`

Drop-in replacement using CUDA streams instead of ThreadPoolExecutor. Adds `_decode_step`
and `_sample_tokens` helpers to keep CC low. Outputs `primary_log_probs`, `tandem_log_probs`,
`model_mask` — matching the production interface. Uses `TokenSelectionStrategy` from the
production recipe.

Benchmark: ~1.4-1.5x speedup over `TandemRollout` (see `model_spec/results/2026-03-04_rollout-stream-opt.md`).

---

### `tandem_rollout_vllm.py` — `TandemRolloutVLLM`

Uses two vLLM `LLM` instances (one per GPU). Step-by-step generation: each step calls
`vllm.generate(max_tokens=1, logprobs=20)` on both models, reconstructs sparse logits from
top-20 logprobs, samples, and Bernoulli-selects. Sequential token appending to `current_token_ids`.

Limitation: re-initializes vLLM inference from scratch per step (no persistent KV cache reuse
across steps within vLLM — relies on prefix caching).

---

### `tandem_rollout_ray.py` — `TandemRolloutRay`

Hot model runs locally; frozen model is a `FrozenModelActor` (Ray remote). Forward calls are
dispatched via ThreadPoolExecutor. The Ray actor manages its own KV cache keyed by `batch_id`,
cleared after each minibatch.

---

### `tandem_rollout_shared_kv.py` — `TandemRolloutSharedKV`

Experimental: attempts to share vLLM's GPU KV cache between two `LLM` instances by
monkey-patching `worker.cache_engine[0].gpu_cache`. Uses vLLM v0 API (`VLLM_USE_V1=0`).
Text-based prompt passing (decode → re-encode each step).

---

### `tandem_worker.py` — `TandemActorRolloutWorker`

Extends `ActorRolloutRefWorker`. Lazy-initializes `TandemRollout` on first `generate_sequences`
call. Delegates to `super()` when tandem is disabled.

---

### `frozen_model_actor.py` — `FrozenModelActor`

Ray remote actor wrapping a frozen HF model on a dedicated GPU. Maintains per-batch KV cache
dict. Used by `TandemRolloutRay`.

---

## 7. verl `ray_trainer.py` — Key Concepts

The upstream `RayPPOTrainer` in `verl/verl/trainer/ppo/ray_trainer.py` is the base class
that any tandem trainer would extend. Key functions:

- `compute_response_mask(data)`: extracts response portion of attention mask
- `compute_advantage(data, adv_estimator, ...)`: computes GAE/GRPO/REINFORCE++ advantages
- `apply_kl_penalty(data, kl_ctrl)`: KL divergence penalty on token-level rewards
- `RayPPOTrainer.fit()`: main training loop — generates sequences, computes rewards,
  computes advantages, updates actor/critic
- The `fit()` loop calls `generate_sequences` -> `compute_ref_log_prob` -> `compute_reward`
  -> `compute_advantage` -> `update_actor` -> `update_critic` per step

The tandem recipe in production (`verl-tandem`) overrides `fit()` to monkey-patch
`compute_response_mask` (for `own_tokens` loss scoping) and `_update_actor` (for
tandem model updates). The scratch prototypes here do not yet have a trainer override.

---

## 8. Production vs Scratch — Feature Gap

| Feature | Production (`verl-tandem/recipe/tandem/`) | Scratch (`verl-llm-tandem/scratch/tandem/`) |
|---------|-------------------------------------------|---------------------------------------------|
| TokenSelectionStrategy (Bernoulli/Chunk/Alternating) | Yes | Only in `tandem_rollout_optimized.py` |
| `primary_log_probs` / `tandem_log_probs` output | Yes | Only in `tandem_rollout_optimized.py` |
| Naming convention (`primary`/`tandem`) | Yes | Uses `model_a`/`model_b` |
| FSDP integration | Full (`TandemActorRolloutRefWorker`) | Partial (`TandemActorRolloutWorker`) |
| Inference-backed path (vLLM async) | Yes (`InferenceBackedTandemActorRolloutRefWorker`) | No |
| TandemTrainer (loss scope, tandem update) | Yes | No |
| CUDA stream optimization | Yes (in `TandemRolloutOptimized`) | Yes (same file) |
| vLLM step-by-step rollout | No | Yes (`tandem_rollout_vllm.py`) |
| Ray-based frozen model | No | Yes (`tandem_rollout_ray.py`) |
| Shared KV cache experiment | No | Yes (`tandem_rollout_shared_kv.py`) |
