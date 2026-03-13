# Test Results — tandem-sampler
**Date**: 2026-03-12
**Module**: vllm_source/vllm/v1/sample/tandem_sampler.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | prob_primary=1.0 greedy output matches single-model (no tandem) greedy output | PASS |
| 2 | prob_primary=0.0 uses frozen model only, deterministic across two runs | PASS |
| 3 | prob_primary=0.5 (bernoulli) generates tokens for multi-prompt batch | PASS |
| 4 | alternating strategy generates tokens, deterministic greedy | PASS |

## Notes
- TandemSampler samples from both primary and frozen logits independently via vLLM's native Sampler, then selects per-token via strategy
- Three strategies implemented: bernoulli (random per-token), chunk (fixed-length alternation), alternating (step-parity)
- model_mask (int32 tensor) output tracks which model authored each token (1=primary, 0=frozen)
- prob_primary=1.0 produces identical token sequences to single-model vLLM — confirms no interference from tandem machinery
- prob_primary=0.0 with greedy decoding is deterministic — confirms frozen model forward is stable
- Tests run via subprocess to isolate CUDA context per test case (v1 engine runs in subprocess)
- All 4 tests passed in 114s total (each spawns a full LLM instance with Qwen3-0.6B on 2 GPUs)
