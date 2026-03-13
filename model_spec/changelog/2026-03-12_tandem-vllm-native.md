# Changelog — tandem-vllm-native
**Date**: 2026-03-12

## Goal
Replace HF-based tandem rollout (slow, OOMs at 512 tokens) with vLLM-native tandem generation.
Both primary (hot) and frozen models use vLLM's paged attention, flash attention, and KV cache.
Primary on cuda:0, frozen on cuda:1. Editable vLLM source at `vllm_source/`.

---

## 8-Phase Roadmap

### Phase 1: TandemConfig — DONE
Add tandem configuration to vLLM's config system.
- `TandemConfig` dataclass: enabled, frozen_model, frozen_gpu_devices, prob_primary, selection_strategy, frozen_quantization, frozen_max_model_len, frozen_model_revision
- Wired into `EngineArgs` → `VllmConfig` via `create_tandem_config()`
- Mutual exclusion with speculative decoding
- **Test**: 17/17 passed (config creation, defaults, validation, propagation, mutual exclusion)
- **Result**: model_spec/results/2026-03-12_tandem-config.md

### Phase 2: TandemModelManager — DONE
Dual model loading onto respective GPUs.
- `TandemModelManager` class in `vllm/v1/worker/tandem.py`
- Frozen model loaded via vLLM's own `_initialize_model()` with `prefix="tandem_frozen."` (NOT HF AutoModel — your feedback corrected this)
- Frozen attention layers merged into primary's `static_forward_context`
- `eval()` + `requires_grad_(False)` on frozen model
- Integrated into `GPUModelRunner.load_model()`
- **Test**: 7/7 passed (both models load, correct devices, frozen params frozen, log verification)
- **Result**: model_spec/results/2026-03-12_tandem-model-manager.md

### Phase 3: Dual Forward Pass — DONE
Both models run forward per decode step in `GPUModelRunner.execute_model()`.
- Frozen KV cache allocated on frozen device, same block count/spec as primary
- `get_kv_cache_spec()` filters `tandem_frozen.*` layers from primary memory budget
- `frozen_forward()` adapts attn_metadata tensors to frozen device, runs frozen model, computes logits
- Both models share same attn_metadata (slot_mapping, block_table, positions) since they process identical sequences
- **Test**: 4/4 passed (boot, single prompt, multi-prompt batch, deterministic greedy)
- **Result**: model_spec/results/2026-03-12_tandem-dual-forward.md

### Phase 4: TandemSampler — DONE
Selection strategy using both models' logits to pick tokens.
- `TandemSampler` in `vllm/v1/sample/tandem_sampler.py`
- Samples from primary_logits and frozen_logits independently via vLLM's native Sampler
- Selects per-token via strategy: bernoulli (random), chunk (fixed-length blocks), alternating (step-parity)
- Outputs `SamplerOutput` + `model_mask` (int32 tensor, 1=primary, 0=frozen)
- Integrated into `GPUModelRunner.execute_model()` when tandem enabled
- **Test**: 4/4 passed (prob=1.0 matches single model, prob=0.0 deterministic, prob=0.5 generates, alternating works)
- **Result**: model_spec/results/2026-03-12_tandem-sampler.md

### Phase 5: KV Cache Sync — DONE
Verified both models see identical token history regardless of which model authored each token.
- No code changes needed — KV cache sync is inherent to the architecture
- Key invariant: scheduler feeds the same chosen token to both models on every step, both use same attn_metadata (slot_mapping, block_table, positions), each writes to its own KV cache tensors
- Tested by exploiting same-model invariant: with identical primary/frozen weights + greedy decoding, output must match single-model output for any prob_primary or strategy
- **Test**: 4/4 passed (frozen-only matches single, mixed matches single, 128-token no drift, multi-prompt isolation)
- **Result**: model_spec/results/2026-03-12_tandem-kv-cache-sync.md

### Phase 6: verl Integration — DONE
Propagated `model_mask` through the full vLLM output chain to verl's `DataProto.batch`.
- Added `tandem_model_mask` field to: `ModelRunnerOutput`, `EngineCoreOutput`, `CompletionOutput`
- Wired `tandem_model_mask` in: `gpu_model_runner.py` (convert tensor→list), `scheduler.py` (per-request extraction), `output_processor.py` (accumulate per-step, pass to CompletionOutput)
- verl edit: `vllm_rollout_spmd.py` (~12 lines) extracts mask from `CompletionOutput`, pads to response_length, adds to `DataProto.batch["model_mask"]` as float32 tensor
- Config injection via `engine_kwargs.vllm.tandem_config` in YAML — zero verl config system edits
- All edits additive with None defaults — non-tandem codepaths completely untouched
- **Test**: 5/5 passed (mask present, mask absent without tandem, prob=1 all-primary, prob=0 all-frozen, multi-prompt correct lengths)
- **Result**: model_spec/results/2026-03-12_tandem-verl-integration.md

### Phase 7: Multi-GPU TP Support — DONE
Support tensor parallelism for each model independently.
- Primary: TP across GPUs 0..N-1, Frozen: TP across GPUs N..2N-1
- Added `_FROZEN_TP` global + `initialize_frozen_model_parallel()` + `use_frozen_tp()` context manager in `parallel_state.py` (all additive)
- `init_worker_distributed_environment` calls `initialize_frozen_model_parallel()` when tandem enabled
- `TandemModelManager`: TP-aware device resolution via `_resolve_frozen_device()`, frozen model init + forward wrapped in `_frozen_tp_context()`
- Config validation: `frozen_tensor_parallel_size` auto-matches primary TP, `frozen_gpu_devices` length validated, PP>1 rejected
- Constraint: `frozen_tensor_parallel_size` must equal `tensor_parallel_size` (same TP degree for both models)
- **Test**: 8/8 passed (5 config validation, 2 frozen TP group lifecycle, 1 E2E regression)
- **Result**: model_spec/results/2026-03-13_tandem-tp-support.md

### Phase 8: End-to-End Training Loop Test — PENDING
Full tandem GRPO training run with vLLM-native rollout.
- Qwen3-0.6B primary + frozen on 2+ GPUs
- Run 5-10 training steps, verify: loss decreases, model_mask distribution matches prob_primary, gradients only flow through primary
- Compare throughput vs HF-based tandem training
- Verify verl metrics (wandb) capture tandem-specific stats

---

## Files Modified/Created

| File | What |
|---|---|
| vllm_source/vllm/config.py | TandemConfig dataclass + tandem_config field on VllmConfig |
| vllm_source/vllm/engine/arg_utils.py | tandem_config on EngineArgs, create_tandem_config(), mutual exclusion with spec decoding |
| vllm_source/vllm/v1/worker/tandem.py | TandemModelManager: frozen model loading, KV cache init, frozen forward, attn metadata adaptation |
| vllm_source/vllm/v1/worker/gpu_model_runner.py | tandem_manager init, frozen loading in load_model(), frozen KV cache in initialize_kv_cache(), frozen forward in execute_model(), frozen layer filtering in get_kv_cache_spec() |
| vllm_source/vllm/v1/sample/tandem_sampler.py | TandemSampler: dual-logit sampling with bernoulli/chunk/alternating strategies, model_mask output |
| vllm_source/vllm/v1/outputs.py | Added tandem_model_mask field to ModelRunnerOutput |
| vllm_source/vllm/v1/engine/__init__.py | Added new_tandem_model_mask field to EngineCoreOutput |
| vllm_source/vllm/v1/core/sched/scheduler.py | Extract per-request tandem_model_mask, pass to EngineCoreOutput |
| vllm_source/vllm/outputs.py | Added tandem_model_mask field to CompletionOutput |
| vllm_source/vllm/v1/engine/output_processor.py | Accumulate tandem_model_mask in RequestState, pass to CompletionOutput |
| verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py | Extract model_mask from CompletionOutput, pad, add to DataProto.batch (verl source edit, `# [MODIFIED]` marked) |
| vllm_source/vllm/distributed/parallel_state.py | Added `_FROZEN_TP`, `initialize_frozen_model_parallel()`, `get_frozen_tp_group()`, `use_frozen_tp()` (all additive, no upstream modification) |
| vllm_source/vllm/v1/worker/gpu_worker.py | Call `initialize_frozen_model_parallel()` after TP init when tandem enabled (`# [MODIFIED]` marked) |

## Benchmark Results

| Method | 64 tok (tok/s) | 512 tok (tok/s) | Notes |
|---|---|---|---|
| HF Tandem (scratch prototype) | 92 | OOM | ThreadPoolExecutor, HF KV cache |
| vLLM Tandem (ours, sequential) | 201 | 207 | 2.2x faster than HF, 0.5x of single (theoretical limit) |
| vLLM Single (reference) | 381 | 414 | No tandem overhead |

## Key Design Decisions
1. Frozen model is vLLM-native (not HF) — uses `_initialize_model()` with prefix="tandem_frozen."
2. Both models share attn_metadata (same sequences) but write to separate KV cache tensors
3. v1 engine (not v0) — verl uses v1 by default
4. Primary on cuda:0, frozen on cuda:1
5. Sequential frozen forward for now (2x latency) — CUDA stream overlap is future optimization

## Open Issues / TODOs
- Prefix caching: frozen model's KV blocks not tracked by scheduler's prefix cache logic. Disable prefix caching when tandem enabled, or mirror primary cache management.
- Sequential frozen forward: currently 2x latency. Overlap with CUDA streams for ~1x latency (future Phase 3b).
- Multi-step scheduling: untested with tandem. May need frozen model to process multiple steps in batch.
