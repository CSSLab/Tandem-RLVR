# Changelog — rollout-stream-opt
**Date**: 2026-03-04

## Added
- `scratch/tandem/tandem_rollout_optimized.py`: `TandemRolloutOptimized` — CUDA stream parallelism in decode loop, CC-compliant helper extraction (`_decode_step`, `_sample_tokens`). Output interface matches production `recipe/tandem/tandem_rollout.py`.

## Changed
- Nothing in production code touched this session.

## Removed / commented out
- `scratch/tandem/_test_rollout_speed.py`: deleted after benchmark confirmed passing (per model spec S3).

## Open issues / TODOs
- `TandemRolloutOptimized` should replace `TandemRollout` in production after multi-GPU FSDP integration test.
- CUDA stream synchronization pattern needs verification for same-device case.
