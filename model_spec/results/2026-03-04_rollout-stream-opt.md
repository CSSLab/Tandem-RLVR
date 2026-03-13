# Test Results — rollout-stream-opt
**Date**: 2026-03-04
**Module**: `scratch/tandem/tandem_rollout_optimized.py`
**Baseline**: `scratch/tandem/tandem_rollout.py` (legacy HF-loop)
**Hardware**: 2x GPU (cuda:0 = model A / primary, cuda:1 = model B / tandem), Qwen3-0.6B both sides

## Cases

| # | Description | Status |
|---|-------------|--------|
| 1 | batch=1, T=1 (single decode step edge case) — shape check + single-step token shape assert | PASS |
| 2 | batch=1, T=64 — output shapes correct, primary_log_probs finite and <= 0 | PASS |
| 3 | batch=4, T=64 — output shapes correct, primary_log_probs finite and <= 0 | PASS |

## Benchmark Results (run 2, post-CC refactor)

| Case | Legacy (s) | Optimized (s) | Speedup | tok/s (opt) |
|------|-----------|--------------|---------|-------------|
| batch=1, T=1 | 0.078 | 0.068 | 1.15x | 14.7 |
| batch=1, T=64 | 5.897 | 4.002 | **1.47x** | 16.0 |
| batch=4, T=64 | 5.785 | 4.094 | **1.41x** | 62.5 |

## Notes

- ~1.4-1.5x speedup from CUDA stream parallelism in decode loop.
- CC of `generate_sequences` reduced to <= B by extracting `_decode_step` and `_sample_tokens`.
- Legacy rollout does not output `primary_log_probs`/`tandem_log_probs`; optimized version does.
