# Install

You are building one conda environment holding two editable forks: vLLM at release tag v0.19.1 plus
`third_party/vllm-tandem.patch`, and verl at `cbd7f9f4` plus `third_party/verl-tandem.patch`. Both
patches are pure Python. No CUDA code is compiled.

## Host requirements

- At least 2 NVIDIA GPUs, A100 or H100 class. The reference machine was 2 x A100 80GB. The tandem
  arm needs both cards: the senior trains and serves rollouts on one, the junior's weights and KV
  cache sit on the other.
- NVIDIA driver 525.60.13 or newer. The stack is built for CUDA 12.8 and runs on older CUDA 12
  drivers through minor version compatibility. The reference driver reported CUDA 12.2.
- conda, roughly 60 GB of disk for the environment and the models.
- Network access for the initial pulls: two git clones, the pip wheels, and the Hub models.

## Pinned versions

| package | version |
|---|---|
| python | 3.11 |
| torch | 2.10.0+cu128, pulled by vLLM |
| vllm | release tag `v0.19.1`, commit `b1388b1fbf5aaef47937fabe98931211684666a6`, plus the tandem patch |
| transformers | 5.5.4 |
| verl | `0.9.0.dev` at commit `cbd7f9f462c0230f4f6161462b5b294c9d55d453`, plus the tandem patch |
| TransferQueue | 0.1.8 |
| ray | 2.55.1 |
| tensordict | 0.10.0 |

`env/requirements-freeze.txt` is the full 227 package pin from the machine that produced the runs.
Treat it as the tie breaker if anything below drifts. The base commits are also in
`third_party/VLLM_BASE_COMMIT.txt` and `third_party/VERL_BASE_COMMIT.txt`, which is where the
scripts read them from.

`transformers` is pinned rather than left to resolve. vLLM 0.19.1 will pull a much newer version,
which was not the version these runs used.

## 1. Environment

```bash
bash env/install.sh /path/to/envs/tandem-verl
export TANDEM_ENV=/path/to/envs/tandem-verl
```

That creates a python 3.11 environment, installs `vllm==0.19.1` (which brings torch 2.10.0+cu128)
and pins `transformers==5.5.4`. Installing the vLLM release wheel first is deliberate: the forked
source is installed over it in the next step, and the wheel's compiled extensions are reused.

## 2. Both forks

```bash
bash third_party/apply_patches.sh --install
```

The script clones `vllm-project/vllm` and `volcengine/verl` into `third_party/vllm` and
`third_party/verl`, checks out the two pinned commits, verifies that `HEAD` is the commit the pin
file names, and applies the two patches. It is idempotent: a tree that already carries its patch is
left alone. With `--install` it then runs the two pip installs described below. Without the flag it
stops after patching, and you run them yourself.

Point it at a different python with `PYTHON=/path/to/bin/python`, and at different clone locations
with `VLLM_SRC` and `VERL_SRC`.

### vLLM, editable, reusing the compiled extensions

```bash
VLLM_USE_PRECOMPILED=1 $TANDEM_ENV/bin/python -m pip install -e third_party/vllm --no-build-isolation
$TANDEM_ENV/bin/python -c "import vllm; from vllm import _C; print('vllm', vllm.__version__)"
```

`VLLM_USE_PRECOMPILED=1` skips the CUDA build and reuses the binary extensions. If the import of
`_C` fails, the editable tree is missing them, and the fix is to take them from the release wheel
already installed in step 1. They are ABI stable and the wheel is the same v0.19.1 the source tree
is checked out at:

```bash
$TANDEM_ENV/bin/python -m pip download vllm==0.19.1 --no-deps -d /tmp/vw
$TANDEM_ENV/bin/python -c "import zipfile,glob; zipfile.ZipFile(glob.glob('/tmp/vw/vllm-*.whl')[0]).extractall('/tmp/vw/x')"
cp /tmp/vw/x/vllm/*.so third_party/vllm/vllm/
cp /tmp/vw/x/vllm/vllm_flash_attn/*.so third_party/vllm/vllm/vllm_flash_attn/
```

Eight extensions are needed: `_C`, `_moe_C`, `_C_stable_libtorch`, `_flashmla_C`,
`_flashmla_extension_C`, `cumem_allocator`, `vllm_flash_attn/_vllm_fa2_C` and
`vllm_flash_attn/_vllm_fa3_C`, all `.abi3.so`.

### verl, editable, base install only

```bash
$TANDEM_ENV/bin/python -m pip install -e third_party/verl -c env/constraints.txt
$TANDEM_ENV/bin/python -m pip install "TransferQueue==0.1.8"
```

Install the base package, not the `[vllm]` extra. verl's base requirements pin neither vllm nor
torch, so this leaves the stack from step 1 intact. The `[vllm]` extra pins `vllm<=0.12.0` and would
downgrade the fork out from under you. `env/constraints.txt` holds torch, vllm and transformers at
the versions above as a second line of defence. TransferQueue is a dependency of verl 0.9's v1
trainer and is not installed automatically.

## 3. Smoke check

In order. Each step is cheap and each one fails in a distinguishable way.

```bash
# a. the stack imports and can talk to the GPUs
$TANDEM_ENV/bin/python env/smoke.py

# b. both forks import, including the tandem modules the patches add
$TANDEM_ENV/bin/python -c "from vllm.config import TandemConfig; from vllm.v1.sample.tandem_sampler import TandemSampler; print('vllm fork OK')"
$TANDEM_ENV/bin/python -c "import verl, transfer_queue; from verl.workers.utils.losses import _apply_tandem_senior_gate; print('verl fork OK')"

# c. CPU tests: the word handoff schedule, and the gate zeroing junior positions
$TANDEM_ENV/bin/python -m pytest tests/test_word_state_machine.py tests/test_senior_gate.py

# d. GPU test: a real dual decoder rollout produces a mask of the right length,
#    with roughly half the tokens attributed to the senior
```

Step (b) is the one that catches the failure mode that matters. If the vLLM import succeeds but
`TandemSampler` is missing, vLLM is installed from the release wheel rather than the patched tree,
no authorship mask will ever be produced, and training will run as ordinary GRPO without complaint.
See the gate description in `docs/ARCHITECTURE.md`.

The GPU tests default to `Qwen/Qwen3-0.6B` and read `TANDEM_TEST_MODEL` for an override. See
`tests/README.md` for what each one proves.

## 4. Site configuration

Copy `train/env.sh.example` to `train/env.sh` and fill it in. It is the only file with paths specific
to your machine, it is listed in `.gitignore`, and all four training launchers and `eval/run_all.sh`
source it.

## Regenerating the patches

pinned base commits and rewrites the two patch files. The script's comments record which hunks are
deliberately excluded.
