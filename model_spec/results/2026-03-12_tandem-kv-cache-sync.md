# Test Results — tandem-kv-cache-sync
**Date**: 2026-03-12
**Module**: vllm_source/vllm/v1/worker/tandem.py, vllm_source/vllm/v1/worker/gpu_model_runner.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | prob=0.0 (frozen-only) greedy matches single-model greedy output | PASS |
| 2 | prob=0.5 bernoulli (mixed selection) greedy matches single-model greedy output | PASS |
| 3 | 128-token alternating strategy greedy matches single-model (no KV drift over long sequence) | PASS |
| 4 | 3 prompts of different lengths, prob=0.5, all match single-model per-request output | PASS |

## Test Design Rationale
Both primary and frozen models are Qwen3-0.6B (identical weights). With greedy decoding (temperature=0), both models always produce the same argmax token regardless of which model authored the previous token. Therefore, if KV caches are correctly synchronized, tandem output must be identical to single-model output for any prob_primary value or selection strategy.

Any divergence would prove KV cache corruption — one model's cache failing to incorporate the token chosen by the other model.

## Invariant Verified
At each decode step t, both models have processed the identical token sequence [0, 1, ..., t-1]. This holds because:
1. TandemSampler outputs one chosen token per position
2. That token flows back to the scheduler as ModelRunnerOutput
3. Scheduler creates new_token_ids and feeds them as input_ids to both models on the next step
4. Both models share the same attn_metadata (slot_mapping, block_table, positions) since they process the same sequences
5. Each model writes to its own KV cache tensors using the same slot mapping

## Notes
- No code changes were needed for Phase 5 — KV cache sync is inherent to the architecture
- Cross-device (cuda:0 → cuda:1) float arithmetic is not bit-identical, but argmax is robust to tiny differences
- Long sequence test (128 tokens) confirms no accumulated drift in KV cache over many decode steps
- Multi-prompt test confirms per-request KV cache isolation — different-length prompts don't interfere
