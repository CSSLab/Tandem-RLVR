# What the patches contain

Tandem rollout is a change to two projects. `apply_patches.sh` clones each at the commit pinned in
`VLLM_BASE_COMMIT.txt` and `VERL_BASE_COMMIT.txt` and applies the corresponding patch. Both were
checked with `git apply --check` against a tree extracted at those commits.

## vLLM, base `b1388b1f` (release tag v0.19.1)

14 files, 815 lines. Three are new and hold the mechanism:

- `vllm/config/tandem.py`, the configuration object and its validation.
- `vllm/v1/sample/tandem_sampler.py`, the handoff schedule. It decides which model emits each
  token and records the choice.
- `vllm/v1/worker/tandem.py`, the second model: loading it on its own device, mirroring the
  senior's attention metadata, and giving it its own KV cache.

The other eleven are the hooks that carry the per token authorship record from the sampler out
through `CompletionOutput`.

## verl, base `cbd7f9f4`

13 files, 160 lines. The one that matters is
`verl/workers/utils/losses.py::_apply_tandem_senior_gate`, which multiplies the authorship mask
into the response mask before the policy gradient sum. The rest carries the mask from the rollout
into the training batch, adds two configuration keys, forces validation to the senior alone, and
persists Hugging Face weights per save.

## Maintenance

The patches apply at the pinned commits only; upstream has moved past both, and a later base needs the patches rebased.
