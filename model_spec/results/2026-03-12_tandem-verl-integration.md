# Test Results — tandem-verl-integration
**Date**: 2026-03-12
**Module**: vllm output chain (6 files) + verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | model_mask present in CompletionOutput with tandem enabled, correct length and values (0/1 only) | PASS |
| 2 | model_mask absent (None) when tandem is disabled — no interference with standard vLLM | PASS |
| 3 | prob_primary=1.0 produces all-primary mask (every value = 1) | PASS |
| 4 | prob_primary=0.0 produces all-frozen mask (every value = 0) | PASS |
| 5 | Multi-prompt batch: each prompt's mask length matches its token count, all values valid | PASS |

## Propagation Chain Verified
```
GPUModelRunner.execute_model()  →  tandem_model_mask (torch.Tensor)
    ↓ convert to list[list[int]]
ModelRunnerOutput.tandem_model_mask
    ↓ per-request extraction
Scheduler.update_from_output()  →  EngineCoreOutput.new_tandem_model_mask
    ↓ msgspec serialization over ZMQ
OutputProcessor.process_outputs()  →  RequestState.tandem_model_mask (accumulated)
    ↓
CompletionOutput.tandem_model_mask  →  list[int], one per generated token
    ↓
verl vLLMRollout.generate_sequences()  →  DataProto.batch["model_mask"]
```

## Notes
- All vLLM edits are additive: new optional fields with None defaults. Non-tandem codepaths untouched.
- EngineCoreOutput uses omit_defaults=True in msgspec, so new_tandem_model_mask adds zero serialization overhead when tandem is disabled.
- verl edit is minimal (~12 lines) and guarded by `getattr(..., 'tandem_model_mask', None)` and `if tandem_model_masks:` — standard rollout unaffected.
- Config injection via engine_kwargs (no verl config system edits needed):
  ```yaml
  actor_rollout_ref:
    rollout:
      engine_kwargs:
        vllm:
          tandem_config: {enabled: true, frozen_model: "...", ...}
  ```
