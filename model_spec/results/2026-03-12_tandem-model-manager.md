# Test Results — tandem-model-manager
**Date**: 2026-03-12
**Module**: vllm_source/vllm/v1/worker/tandem.py, vllm_source/vllm/v1/worker/gpu_model_runner.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | TandemModelManager import | PASS |
| 2 | get_frozen_model before load raises RuntimeError | PASS |
| 3 | get_frozen_device before load raises RuntimeError | PASS |
| 4 | Disabled config skips load entirely | PASS |
| 5 | Frozen config inherits primary model settings | PASS |
| 6 | Frozen config supports different model path | PASS |
| 7 | Full LLM boot with tandem: both models loaded, frozen on cuda:1 (subprocess verified) | PASS |

## Notes
- Frozen model loaded via vLLM's own `_initialize_model()` + `load_weights()` — same vLLM model architecture as primary
- Uses `prefix="tandem_frozen."` to avoid attention layer name collisions in `static_forward_context`
- v1 engine runs in a subprocess — integration test uses subprocess to verify log output
- Both models are vLLM-native: frozen model gets same optimizations (paged attention, flash attention) as primary
- Initial HF AutoModelForCausalLM approach was abandoned (defeats purpose of vLLM-native speed)
- Initial vLLM `get_model()` approach failed on duplicate layer names — fixed with prefix namespacing
