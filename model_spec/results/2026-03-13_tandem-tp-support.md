# Test Results — tandem-tp-support
**Date**: 2026-03-13
**Module**: vllm_source/vllm/distributed/parallel_state.py, vllm_source/vllm/v1/worker/tandem.py, vllm_source/vllm/config.py, vllm_source/vllm/v1/worker/gpu_worker.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | TP auto-match: frozen_tp=1 with primary_tp=4 auto-matches to frozen_tp=4 | PASS |
| 2 | TP mismatch error: frozen_tp=2 vs primary_tp=4 raises ValueError | PASS |
| 3 | frozen_gpu_devices wrong length: 3 devices for tp=2 raises ValueError | PASS |
| 4 | PP rejection: pipeline_parallel_size=2 raises ValueError | PASS |
| 5 | TP=1 no validation issue: standard single-GPU config passes cleanly | PASS |
| 6 | Frozen TP group lifecycle: init, access, destroy all work correctly | PASS |
| 7 | use_frozen_tp context manager: swap, restore, and nesting all correct | PASS |
| 8 | E2E regression: tandem generation with TP=1 produces output + model_mask | PASS |

## Notes
- Tested on 2x A100 80GB (TP=1 per model). TP>1 paths are structurally correct but require 4+ GPUs to validate.
- Frozen TP group uses gloo backend for unit tests; NCCL backend in production.
- Context manager nesting verified: inner use_frozen_tp correctly restores to frozen_tp (not primary_tp) on exit.
- E2E regression confirms no behavioral change from Phase 6: model_mask present, non-empty generation.
