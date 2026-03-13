# Test Results — tandem-config
**Date**: 2026-03-12
**Module**: vllm_source/vllm/config.py, vllm_source/vllm/engine/arg_utils.py

## Cases
| # | Description | Status |
|---|-------------|--------|
| 1 | TandemConfig basic creation with all fields | PASS |
| 2 | TandemConfig.from_dict with custom GPU devices and TP | PASS |
| 3 | All three selection strategies (bernoulli/chunk/alternating) | PASS |
| 4 | Disabled config does not require frozen_model | PASS |
| 5 | compute_hash deterministic and changes on param change | PASS |
| 6 | Enabled without frozen_model raises ValueError | PASS |
| 7 | prob_primary out of [0,1] raises ValueError | PASS |
| 8 | prob_primary boundary values 0.0 and 1.0 accepted | PASS |
| 9 | Invalid selection_strategy raises ValueError | PASS |
| 10 | Unknown kwargs raise TypeError | PASS |
| 11 | EngineArgs propagates tandem_config to VllmConfig with target configs | PASS |
| 12 | None tandem_config stays None in VllmConfig | PASS |
| 13 | Tandem + speculative together raises error | PASS |
| 14 | VllmConfig dataclass has tandem_config field | PASS |

## Notes
- 17 tests total, all passing
- TandemConfig follows same pattern as SpeculativeConfig (dataclass, from_dict, compute_hash)
- Mutual exclusion with speculative decoding enforced at EngineArgs level
- v1 engine oracle rejects speculative before our check runs, but both paths error correctly
