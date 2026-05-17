# Terminology — Paper ↔ Code

**Project**: `verl-llm-tandem` · Tandem Reinforcement Learning (TRL)
**Authority**: This file is the single source of truth for naming. The forthcoming sweeping rename (SPEC §2 procedure, batched as one commit) converges every identifier in tracked code to the **target** column below. Until that commit lands, the codebase still uses the **legacy** spellings — both columns are valid to read; only the target column is valid to write.

---

## 1. Naming policy at a glance

| Paper concept | Legacy spelling(s) in code | Target spelling | Rationale |
|---|---|---|---|
| Trainable model π_sen | `primary`, `hot`, `model_a`, `target_*` (in TandemConfig) | `senior` | Match paper §3.2 |
| Frozen partner π_jun | `frozen`, `model_b` | `junior` | Match paper §3.2 |
| Per-token authorship indicator (1=senior, 0=junior) | `tandem_model_mask`, `model_mask`, `model_a_mask` | `authorship_mask` | Match paper §A.1.1 "authorship stream"; eliminates 3-way ambiguity |
| Tandem Reinforcement Learning (the method) | "TT" (tandem training), "tandem native GRPO" | **TRL** | Match paper title and Algorithm 1 |
| Senior-handoff probability (Bernoulli p) | `prob_primary` | `prob_senior` | Mirror rename of primary→senior |
| Subword-span cap (paper K=32) | `max_gap_tokens` | `max_gap_tokens` *(unchanged)* | Already paper-aligned in spirit; rename adds no clarity |
| Junior-token loss-weight coefficient | `tandem_jr_tkn_weight` | `junior_token_loss_weight` | Paper λ_jun is a different mechanism (auxiliary KL term), but reaches the same fixed point at 0; renamed to reflect what it *actually* does in our code (response-mask weighting) |
| Word-boundary handoff strategy | `selection_strategy="word"` | unchanged | Already paper-aligned |

---

## 2. Detailed rename map

### 2.1 Module-level constants & functions

| Legacy | Target | File |
|---|---|---|
| `FROZEN_PREFIX = "tandem_frozen."` | `JUNIOR_PREFIX = "tandem_junior."` | `vllm_source/vllm/v1/worker/tandem.py` |
| `is_frozen_layer(name)` | `is_junior_layer(name)` | `vllm_source/vllm/v1/worker/tandem.py`; callers in `gpu_model_runner.py` |
| `get_frozen_tp_group()` | `get_junior_tp_group()` | `vllm_source/vllm/distributed/parallel_state.py`; callers in `gpu_worker.py`, `tandem.py` |
| `use_frozen_tp()` | `use_junior_tp()` | same as above |
| `_frozen_tp_context()` (helper) | `_junior_tp_context()` | `vllm_source/vllm/v1/worker/tandem.py` |
| `TandemModelManager.frozen_forward()` | `TandemModelManager.junior_forward()` | `vllm_source/vllm/v1/worker/tandem.py`; caller in `gpu_model_runner.py` |
| `TandemModelManager.load_frozen_model()` | `.load_junior_model()` | same |
| `TandemModelManager.initialize_frozen_kv_cache()` | `.initialize_junior_kv_cache()` | same |
| `TandemModelManager.get_frozen_model()` | `.get_junior_model()` | same |
| `TandemModelManager.get_frozen_device()` | `.get_junior_device()` | same |

### 2.2 `TandemConfig` fields (vllm_source/vllm/config.py)

The class is a stable public surface — renaming its fields breaks every run script and every saved checkpoint's config blob. We rename anyway; pre-rename run scripts and checkpoint configs become invalid without manual migration (noted in §5). No backward-compat shim is added.

| Legacy field | Target field |
|---|---|
| `frozen_model` | `junior_model` |
| `frozen_model_revision` | `junior_model_revision` |
| `prob_primary` | `prob_senior` |
| `frozen_tensor_parallel_size` | `junior_tensor_parallel_size` |
| `frozen_gpu_devices` | `junior_gpu_devices` |
| `frozen_gpu_memory_utilization` | `junior_gpu_memory_utilization` |
| `frozen_dtype` | `junior_dtype` |
| `frozen_quantization` | `junior_quantization` |
| `frozen_enforce_eager` | `junior_enforce_eager` |
| `frozen_max_model_len` | `junior_max_model_len` |
| `target_model_config` | `senior_model_config` |
| `target_parallel_config` | `senior_parallel_config` |
| `selection_strategy` *(unchanged)* | `selection_strategy` |
| `boundary_token_ids` *(unchanged)* | `boundary_token_ids` |
| `chunk_size` *(unchanged)* | `chunk_size` |
| `max_gap_tokens` *(unchanged)* | `max_gap_tokens` |
| `enabled` *(unchanged)* | `enabled` |

### 2.3 `TandemSampler` (vllm_source/vllm/v1/sample/tandem_sampler.py)

| Legacy | Target |
|---|---|
| `forward(primary_logits, frozen_logits, ...)` | `forward(senior_logits, junior_logits, ...)` |
| `primary_output`, `frozen_output` (locals) | `senior_output`, `junior_output` |
| `primary_tokens`, `frozen_tokens` | `senior_tokens`, `junior_tokens` |
| `use_primary` (return of `_select`) | `use_senior` |
| `self.prob_primary` (attribute) | `self.prob_senior` |
| `_word_state` cache tuple `(is_senior, ...)` *(unchanged — already paper-aligned)* | same |

### 2.4 `TandemModelManager` (vllm_source/vllm/v1/worker/tandem.py) attribute names

| Legacy | Target |
|---|---|
| `self.frozen_model` | `self.junior_model` |
| `self.frozen_device` | `self.junior_device` |
| `self.frozen_kv_caches` | `self.junior_kv_caches` |
| `self.primary_vllm_config` | `self.senior_vllm_config` |
| `frozen_attn_metadata` (local) | `junior_attn_metadata` |
| `frozen_input_ids`, `frozen_positions`, `frozen_logits_indices` (locals in `frozen_forward`) | `junior_input_ids`, `junior_positions`, `junior_logits_indices` |
| `frozen_sfc`, `primary_sfc` (locals) | `junior_sfc`, `senior_sfc` |
| `frozen_attn_count` (local) | `junior_attn_count` |
| `frozen_layers`, `frozen_kv_caches` (locals in `initialize_frozen_kv_cache`) | `junior_layers`, `junior_kv_caches` |
| `frozen_vllm_config`, `frozen_config`, `frozen_hf_config`, `frozen_local_rank` | `junior_*` (mechanical rename) |

### 2.5 Authorship-mask propagation chain

The mask passes through five files in the vLLM output chain plus two in verl. Every spelling collapses to one target name.

| Legacy field/var | Target | Files |
|---|---|---|
| `tandem_model_mask` | `authorship_mask` | `vllm_source/vllm/outputs.py`, `v1/outputs.py`, `v1/engine/__init__.py`, `v1/engine/output_processor.py`, `v1/core/sched/scheduler.py`, `v1/worker/gpu_model_runner.py`, `v1/sample/tandem_sampler.py` |
| `new_tandem_model_mask` (per-step delta) | `new_authorship_mask` | `v1/engine/__init__.py`, `v1/engine/output_processor.py`, `v1/core/sched/scheduler.py` |
| `tandem_mask` (local var) | `authorship_mask` | same |
| `model_mask` (verl DataProto batch key) | `authorship_mask` | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, `verl/workers/actor/dp_actor.py` |
| `model_a_mask` (legacy verl key from HF-loop scratch path) | dropped (HF-loop scratch path is gitignored R&D; the live verl path uses only the renamed `authorship_mask`) | `dp_actor.py:382` — drop the `model_a_mask` branch entirely |

### 2.6 Loss-weight knob (actor)

| Legacy | Target |
|---|---|
| `tandem_jr_tkn_weight` (config field on `ActorConfig`) | `junior_token_loss_weight` |
| `self.config.get("tandem_jr_tkn_weight", 0.0)` | `self.config.get("junior_token_loss_weight", 0.0)` |
| `+actor_rollout_ref.actor.tandem_jr_tkn_weight=…` in run scripts | `+actor_rollout_ref.actor.junior_token_loss_weight=…` |

Note: the paper's λ_jun (§3.4) is a *different* mechanism — an auxiliary KL term toward π_jun on junior-emitted positions. Our code instead reweights junior-emitted positions in the standard policy-gradient mask. The two coincide at the value 0 (senior-only loss). The renamed `junior_token_loss_weight` reflects the mechanism actually implemented, not the paper's λ_jun. TERMINOLOGY-§3 elaborates.

### 2.7 Metric names (wandb keys)

| Legacy | Target |
|---|---|
| `tandem/primary_token_fraction` | `tandem/senior_token_fraction` |
| `tandem/frozen_token_fraction` | `tandem/junior_token_fraction` |
| `tandem/switches_per_seq` *(unchanged)* | same |
| `tandem/tokens_per_sent` *(unchanged)* | same |

### 2.8 Run-script shell variables

Already paper-aligned (`SENIOR_MODEL`, `JUNIOR_MODEL`) — no rename. Only the hydra-override keys change, mirroring §2.2 and §2.6.

---

## 3. Loss-weight semantics — paper λ_jun vs code `junior_token_loss_weight`

Two different mechanisms that both collapse to the paper's senior-only objective when set to 0.

**Paper (§3.4, eq. 1)**: senior is trained with the GRPO clipped-surrogate objective, where the response mask is narrowed by `m_{j,i}` (1 at senior positions, 0 at junior positions). Junior-emitted tokens contribute no gradient. The paper additionally mentions a *separate* auxiliary loss term — a soft KL pull of π_sen toward π_jun on junior-emitted positions, weighted by coefficient λ_jun — and explicitly sets λ_jun = 0 to keep the per-token loss formally identical to vanilla GRPO.

**Our code (`verl/workers/actor/dp_actor.py:417-434`)**: the policy-gradient loss multiplies the response mask elementwise by a weight vector built from the authorship mask: senior positions get weight 1; junior positions get weight `junior_token_loss_weight`. With `junior_token_loss_weight=0`, this exactly reproduces the paper's senior-only mask. With `junior_token_loss_weight>0`, junior positions contribute to the standard policy-gradient sum at reduced weight — this is **not** the paper's λ_jun (which would be a separate KL term), but a related "soft mask" that interpolates between senior-only (0) and joint (1) GRPO loss.

For paper reproduction: set `junior_token_loss_weight=0`. The off-paper values 0.15 and 0.2 found in `run_tandem_native_grpo_{math,gsm8k}.sh` are R&D-only ablations and are documented to cause a training cliff (`model_spec/results/2026-04-29_tt-math-takeaways.md`).

---

## 4. What is **not** being renamed

These remain on legacy names by design — listed explicitly so the rename pass skips them.

- **`scratch/tandem/` prototypes** (`tandem_rollout.py`, `tandem_rollout_optimized.py`, `tandem_rollout_vllm.py`, `tandem_rollout_ray.py`, `tandem_rollout_shared_kv.py`, `tandem_worker.py`, `frozen_model_actor.py`): R&D history, gitignored via `scratch/` in `.gitignore`. The `model_a`/`model_b` and `frozen_model_actor` naming there is left alone.
- **Strategy literal values** (`"bernoulli"`, `"chunk"`, `"alternating"`, `"sentence"`, `"word"`): user-facing config strings; renaming them would invalidate every run-script invocation and result file. They stay.
- **Module/file names** (`tandem_sampler.py`, `tandem.py`, `TandemSampler`, `TandemModelManager`, `TandemConfig`, `TandemSelectionStrategy`): the `Tandem` prefix is the paper-blessed name of the method, so it stays. Only the *internal* primary/frozen vocabulary is replaced with senior/junior.
- **Upstream vLLM and verl identifiers** we did not author (`SamplerOutput`, `SamplingMetadata`, `DataProto`, etc.): out of scope.
- **`max_gap_tokens`, `chunk_size`, `boundary_token_ids`, `selection_strategy`, `enabled`**: already paper-aligned or paper-neutral.
- **Archived shell scripts** under `archive/` and `verl/verl/archive/`: gitignored, not touched.

---

## 5. Breaking-change implications

The rename pass is a one-shot hard break — no compatibility shims:

1. **Run scripts**: any hydra override referencing `prob_primary`, `frozen_model`, `frozen_gpu_devices`, `tandem_jr_tkn_weight`, etc. must be updated. All six tracked `run_*.sh` scripts in `verl/` will be updated in the same commit as the rename.
2. **Checkpoint configs**: pre-rename checkpoints under `scratch/checkpoints/` have `tandem_config` blobs serialized with the legacy field names. Reloading them after the rename requires a one-time JSON edit (a small migration helper script will live at `scratch/migrate_tandem_config.py` if needed; out of scope for the rename commit).
3. **wandb dashboards**: any saved view filtering on `tandem/primary_token_fraction` or `tandem/frozen_token_fraction` will need the metric key updated. Historical runs retain the legacy keys; new runs emit the renamed keys.
4. **External code reading our authorship mask** (e.g., the `tandem_eval/` suite): the rollout DataProto key changes from `model_mask` to `authorship_mask`. The eval scripts will be updated in the same commit.

---

## 6. Convention for new code

After the rename commit lands, all newly authored code in this repo MUST:

- Use `senior` / `junior` for the two models, never `primary` / `frozen` / `hot` / `target` / `model_a` / `model_b`.
- Use `authorship_mask` for the per-token authorship indicator, never `model_mask` / `model_a_mask` / `tandem_model_mask`.
- Use `prob_senior` for the Bernoulli handoff parameter (not `prob_primary`).
- Refer to the paper's auxiliary KL coefficient (when discussing it textually) as **λ_jun**; refer to our actual implementation knob as **`junior_token_loss_weight`** and note the semantic distinction (§3) when ambiguity matters.
- Refer to the overall method as **Tandem Reinforcement Learning (TRL)** in user-facing prose; `tandem_*` identifiers and the `Tandem*` class prefix remain.
