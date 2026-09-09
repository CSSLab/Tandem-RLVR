# Environment

One conda environment runs everything in this repo: both training arms, the evaluation
surface and the tests, with both forks installed into it. This file says what is in that
environment and why. `docs/INSTALL.md` is the procedure, and it drives the files here.

## The stack

| Package | Version | Why this version |
|---|---|---|
| python | 3.11 | vLLM 0.19.1 publishes cp311 wheels |
| torch | 2.10.0+cu128 | not chosen; it is what vLLM 0.19.1 resolves |
| vllm | 0.19.1, patched | the base that `third_party/vllm-tandem.patch` applies to |
| transformers | 5.5.4 | the version the recorded runs used; vLLM 0.19.1 on its own resolves a later one |
| verl | 0.9.0.dev0, patched | the base that `third_party/verl-tandem.patch` applies to |
| TransferQueue | 0.1.8 | verl 0.9's v1 trainer needs it and does not pull it in |

## Why the install order matters

vLLM goes in first and alone. Its resolution fixes torch, the CUDA build and most of the
dependency closure, and everything installed after it has to fit inside that closure.
transformers is then pinned back over what vLLM resolved.

verl goes in last, as the base package with `-c env/constraints.txt`, never with the
`[vllm]` extra. That extra pins `vllm>=0.8.5,<=0.12.0` and would downgrade vLLM to a release
the tandem patch does not apply to, at which point the rollout backend is stock vLLM: the
authorship mask never arrives, and training runs as plain GRPO with no error and no warning.
verl's base requirements pin neither vllm nor torch, which is why the base install is safe.

## Files here

| file | what it is |
|---|---|
| `install.sh` | creates the environment and installs the core stack. Takes the conda prefix as its argument, and stops before the forks. |
| `constraints.txt` | the pins passed to the verl install, so that no transitive requirement can move torch, vllm, transformers or numpy. |
| `requirements-freeze.txt` | `pip freeze` of the environment that produced the runs, 233 packages. The tie breaker when a version is in question. |
| `smoke.py` | prints the three versions, where `vllm` and `verl` import from, and the visible GPU count. |

Read the import locations `smoke.py` prints, not only the versions. An import that resolves
to `site-packages` rather than to the patched clone means the fork is not the code being run,
which is the failure described above.
