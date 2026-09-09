# Architecture: how the authorship mask reaches the loss

Two forks cooperate. The vLLM fork runs a frozen junior in lockstep with the trainable senior and
records, per token, which of the two emitted it. The verl fork carries that record into the training
batch and uses it to zero the junior's tokens out of the policy gradient. Neither half is useful
alone: without the vLLM patch no mask is produced, and without the verl patch a mask that is
produced is ignored.

Both patches and their pinned base commits are in `third_party/`. Line references below are to the
patched trees.

## The mask, end to end

| # | Where | What happens |
|---|---|---|
| 1 | `vllm/v1/sample/tandem_sampler.py`, `TandemSampler.forward` | The senior's and the junior's logits are both sampled under the request's own sampling parameters. `_select` returns a boolean row vector `use_primary`, and `torch.where` picks each row's token from the corresponding model. The vector is returned as `model_mask` (int32, 1 = senior, 0 = junior). |
| 2 | `vllm/v1/worker/gpu_model_runner.py`, `_maybe_tandem_sample` | Called from the sampling path before the stock sampler. It stashes the mask on the runner as `self._tandem_mask` and returns the tandem `SamplerOutput`. Returns `None` when tandem is off, which is how the fork stays a no-op for ordinary runs. |
| 3 | `vllm/v1/worker/gpu_model_runner.py`, `_tandem_mask_to_list` | Expands the scalar for each request into a list with one entry per token aligned to that step's sampled tokens, and attaches it to `ModelRunnerOutput.tandem_model_mask`. Under async scheduling the tensor travels on `AsyncGPUModelRunnerOutput` and is expanded after the device copy, so it lines up with the same validated token list. |
| 4 | `vllm/v1/core/sched/scheduler.py`, `Scheduler.update_from_output` | Slices the per request row out and puts it on `EngineCoreOutput.tandem_model_mask`, next to `new_token_ids`. |
| 5 | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs` appends each chunk to `RequestState.tandem_model_mask`, and `make_request_output` slices it the same way it slices logprobs so a delta response carries only the new positions. |
| 6 | `vllm/outputs.py`, `CompletionOutput.tandem_model_mask` | The public surface: `list[int]`, one entry per token in `token_ids`, or `None` when tandem is off. This is the only thing verl needs to see. |
| 7 | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | Reads `final_res.outputs[0].tandem_model_mask` into `extra_fields["tandem_model_mask"]`, the same side channel `routed_experts` and the teacher fields use. |
| 8 | `verl/experimental/agent_loop/agent_loop.py`, `AgentLoopWorker._run_agent_loop` | Pops the field, truncates it to `len(response_ids)`, right pads with zeros to `rollout_config.response_length`, and stores an int64 tensor on `_InternalAgentLoopOutput.tandem_model_mask`. The padding value does not matter: those positions are already zero in `response_mask`. |
| 9 | `verl/experimental/agent_loop/agent_loop.py`, `_postprocess` | Concatenates the per sample tensors into the batch under the key `tandem_model_mask`, alongside `response_mask`. |
| 10 | `verl/workers/utils/losses.py`, `ppo_loss` | `_tandem_mask_fields` adds the key to the field list passed to `data.select(...)`, without which the mask would be dropped before the loss sees it. |
| 11 | `verl/workers/utils/losses.py`, `_apply_tandem_senior_gate` | Combines the mask into `response_mask`, and this is the only place the training objective changes. |

## The gate

```python
def _apply_tandem_senior_gate(config, response_mask, data, metrics):
    tandem_gate = data.get("tandem_model_mask", None)
    if tandem_gate is None:
        return response_mask
    ...
```

Three properties of that function matter.

It is the **entire loss change**. Everything upstream is transport. `ppo_loss` computes the policy
gradient, the entropy term and the KL term from the mask it returns, so gating once here gates all
three.

It **fails open**. A batch with no `tandem_model_mask` gets its response mask back untouched, and
training continues as ordinary GRPO with no error. That is deliberate, so the fork does not break
non-tandem runs, but it means a broken tandem setup looks exactly like a working GRPO run. See the
verification note below.

It has **one knob**, `actor_rollout_ref.actor.tandem_jr_tkn_weight`, declared on `ActorConfig` in
`verl/workers/config/actor.py` and defaulted to 0.0 in `verl/trainer/config/actor/actor.yaml`. At
0.0 the gate is boolean and junior tokens contribute nothing to the gradient. Above 0.0 the mask
becomes a float weight and junior tokens enter the loss scaled by that value. The paper's runs use
0.0. Either way the junior's tokens still shape the reward, because the verifier scores the finished
team response.

The gate logs `actor/tandem_senior_token_frac`, the share of unmasked response tokens the senior
wrote. It is the only external evidence that the mask arrived.

## Verifying the mask is live

`actor/tandem_senior_token_frac` appears in the training metrics only when a mask is in the batch.
Read it on the first logged step:

- absent: no mask reached the loss. Either `VLLM_TANDEM_CONFIG` is unset, or vLLM was installed
  without the patch. Training is running as plain GRPO.
- present and near `prob_primary`: the senior and junior are splitting authorship as configured.
- present but near 0.0 or 1.0: the handoff schedule is degenerate. Check `selection_strategy`,
  `prob_primary` and the boundary id file.

outside [0.3, 0.7], so a broken setup does not consume a full allocation.

## Configuration is entirely environment variables

verl serializes vLLM engine arguments through `EngineArgs.from_cli_args`, which has no
`--tandem-config` flag. The fork therefore reads the configuration from the environment, and there
is no other way to switch tandem on.

### `VLLM_TANDEM_CONFIG`

A JSON object, read by `EngineArgs.create_tandem_config` in `vllm/engine/arg_utils.py` when no
`tandem_config` was passed programmatically, and parsed into the `TandemConfig` dataclass in
`vllm/config/tandem.py`. Unset means tandem is off and the fork behaves as stock vLLM.

This is what `train/tandem_grpo.sh` exports, with its defaults resolved:

```json
{
  "enabled": true,
  "frozen_model": "Qwen/Qwen3-4B-Instruct-2507",
  "selection_strategy": "word",
  "prob_primary": 0.5,
  "max_gap_tokens": 32,
  "frozen_gpu_devices": [1],
  "boundary_token_ids_path": "train/assets/qwen3_word_boundary_ids.json"
}
```

Keys, with the dataclass defaults:

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | Master switch. False loads no junior and emits no mask. |
| `frozen_model` | `null` | The junior, as a Hub id or a local path. Required when enabled. |
| `frozen_model_revision` | `null` | Hub revision for the junior. |
| `selection_strategy` | `"bernoulli"` | One of `bernoulli`, `chunk`, `alternating`, `sentence`, `word`. The paper uses `word`. |
| `prob_primary` | `0.5` | Probability the senior wins a redraw. |
| `max_gap_tokens` | `32` | For `word`: force a redraw after this many tokens with no boundary token. |
| `boundary_token_ids` | `null` | The boundary set, inline. Required by `sentence` and `word` unless the path form is used. |
| `boundary_token_ids_path` | `null` | A JSON file holding that list. The Qwen3 word set is 53,021 ids, past the size an environment variable can carry, so the file form exists for it. |
| `chunk_size` | `1` | For `chunk`: tokens per turn. |
| `frozen_gpu_devices` | `null` | Logical device index per tensor parallel rank for the junior. Unset places the junior at `tp_rank + tp_size`. |
| `frozen_tensor_parallel_size` | `1` | Raised to the senior's TP size automatically, and must match it. |
| `frozen_gpu_memory_utilization` | `0.9` | |
| `frozen_dtype`, `frozen_quantization`, `frozen_enforce_eager`, `frozen_max_model_len` | `null` | Overrides for the junior, otherwise inherited from the senior's config. |

The config validates on construction: an unknown strategy, a missing boundary set for `word` or
`sentence`, `prob_primary` outside [0, 1], a mismatched tensor parallel size, or
`pipeline_parallel_size > 1` all raise. Speculative decoding is rejected in
`create_engine_config`, since the mask is recorded per sampling step.

### `VLLM_TANDEM_ALL_GPUS`

A comma separated list of the job's physical GPU ids, read by `_tandem_extra_visible_devices` in
`verl/workers/rollout/vllm_rollout/vllm_async_server.py`.

verl pins the rollout server actor's visible devices to the GPUs its worker group owns. With
`trainer.n_gpus_per_node=1` that is one card, so a junior placed on the job's second card is
unreachable and the engine dies with an invalid device ordinal. The hook appends the devices in
`VLLM_TANDEM_ALL_GPUS` that the worker group does not own, which makes them addressable by logical
index, which is what `frozen_gpu_devices` uses. It is a strict no-op unless `VLLM_TANDEM_CONFIG` is
also set.

`train/tandem_grpo.sh` sets it to the job's `CUDA_VISIBLE_DEVICES`. The two GPU layout the paper
used is: senior training and rollout engine on the first card, junior weights and KV cache on the
second, requested as `gpu:2` with `trainer.n_gpus_per_node=1`, because the second card belongs to
the job rather than to Ray.

## Where the junior actually runs

`vllm/v1/worker/tandem.py`, `TandemModelManager`, holds the junior. It deep copies the senior's
`VllmConfig`, swaps in the junior's model and device, loads the weights with `requires_grad` off,
and merges the junior's attention layers into the senior's static forward context under the prefix
`tandem_frozen.`, so vLLM's KV cache binding treats them as ordinary layers.

`initialize_frozen_kv_cache` then allocates the junior's paged KV cache on the junior's device using
the senior's block count, block size and cache dtype. On every forward, `frozen_forward` copies the
senior's input ids, positions, attention metadata and slot mappings across to the junior's device and
runs the junior's own forward. The preconditions this arrangement assumes, none of which are checked
at runtime, are listed under Known limitations in `README.md`.

## Word handoff state

`word` is the only strategy with per request state, and it is the strategy the paper uses.
`TandemSampler` keeps `[author_next, tokens_since_redraw]` per request id. After each step,
`_word_post_update` advances the state with the token actually emitted, and redraws the author when
that token is in the boundary set or when `max_gap_tokens` has passed. A request whose state is
missing, because it is new or because it was preempted, has its state rebuilt by replaying the token
prefix in `_word_replay`.

Two properties of this design are load bearing and easy to undo by accident. Read them before
touching the file:

- The state is keyed by request id. Keying by the identity of the token list is wrong, because
  vLLM's persistent batch reuses list objects across steps and the states cross contaminate.
- The state advances on the token chosen in the current step, not by scanning
  `sampling_metadata.output_token_ids`. That field is populated only when penalties are active, and
  under async scheduling the request's own output ids carry `-1` placeholders that defeat boundary
  detection.

`third_party/COMMIT_LOG.md` describes what the patches contain.

## The second channel

`TandemSampler` also reports `tandem_partner_logprob`, the other model's log probability of
whichever token was emitted, and the same plumbing carries it to `CompletionOutput`. The tandem
objective does not read it. It is available for analysis.

## Other fork changes

Four changes in the patches sit outside the mask path:

- **Validation is senior alone.** The agent loop sets `extra_args["tandem_disabled"] = True` on
  validation requests, and the model runner's `_tandem_force_primary` forces those rows to the
  senior. Reported validation pass@1 is therefore the senior's solo score, not the team's. The flag
  is set in both `verl/experimental/agent_loop/agent_loop.py` and
  `verl/trainer/ppo/v1/agent_loop_tq.py`, because `trainer.use_v1` defaults to true and the v1
  runtime uses `AgentLoopWorkerTQ`.
- **`persist_hf_checkpoint`** in `verl/trainer/ppo/v1/trainer_base.py` copies each save's
  `huggingface/` directory to a separate tree, so intermediate weights survive
  `max_actor_ckpt_to_keep` pruning. Enabled with `+trainer.persist_hf_model=True` and
  `actor.checkpoint.save_contents` including `hf_model`.
- **Cross GPU device widening**, described under `VLLM_TANDEM_ALL_GPUS` above.
- **Per user ZeroMQ socket directory** for colocated weight transfer, `/tmp/verl-zmq-$UID`. Upstream
  puts the socket directly in `/tmp` under a Ray job id that is identical for every fresh local
  cluster, so two users on one node collide and the second run dies with `EADDRINUSE`.

