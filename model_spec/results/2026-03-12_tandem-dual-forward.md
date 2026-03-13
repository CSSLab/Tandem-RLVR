# Test Results — tandem-dual-forward
**Date**: 2026-03-12
**Module**: vllm_source/vllm/v1/worker/tandem.py, vllm_source/vllm/v1/worker/gpu_model_runner.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | Boot with tandem: both models load, KV cache initialized, attn layers merged | PASS |
| 2 | Single prompt generates tokens with tandem enabled | PASS |
| 3 | Multi-prompt batch (3 prompts, different lengths) all generate tokens | PASS |
| 4 | Greedy decoding is deterministic across two runs | PASS |

## Notes
- Both primary and frozen models use vLLM-native attention (flash attention, paged KV cache)
- Frozen model's attention layers registered with `tandem_frozen.` prefix to avoid name collisions
- Frozen layers merged into primary's `static_forward_context` after model init so `set_forward_context` can find them
- Frozen KV cache allocated on frozen device (cuda:1) using same block count and spec as primary
- `get_kv_cache_spec()` filters out frozen layers so engine doesn't double-count memory
- `_adapt_attn_metadata()` copies attention metadata tensors to frozen device for cross-GPU forward
- Key bug fixed: frozen model's `_initialize_model` used deepcopy'd config with separate `static_forward_context` — had to merge frozen layers back into primary's context
