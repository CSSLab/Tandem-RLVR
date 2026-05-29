#!/usr/bin/env bash
# Native (no-Docker) reproduction install for Tandem-RLVR.
# Mirrors Dockerfile.repro exactly. Requires: user-space conda, ~15 GB disk, network.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${TANDEM_ENV_NAME:-tandem-verl}"
CONDA_BASE="$(conda info --base)"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[install] creating conda env '${ENV_NAME}' (python 3.10)..."
  conda create -y -n "${ENV_NAME}" python=3.10 pip
else
  echo "[install] reusing existing conda env '${ENV_NAME}'"
fi

conda activate "${ENV_NAME}"

export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[install] installing pinned pip stack (vllm 0.8.5, torch 2.6.0, ...)..."
pip install \
  vllm==0.8.5 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  tensordict==0.6.2 "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
  transformers==4.57.3 accelerate datasets \
  ray[default] codetiming hydra-core wandb dill pybind11 mathruler math-verify \
  "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" "pydantic>=2.9"

echo "[install] flash-attn (prebuilt wheel, no compile)..."
pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo "[install] flashinfer (prebuilt wheel)..."
pip install \
  https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

echo "[install] applying vLLM tandem overlay..."
# vLLM may print platform warnings to stdout on login nodes; take the last line only.
VLLM_DIR="$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)"
/bin/cp -rf "${REPO_ROOT}/vllm_source/vllm/"* "${VLLM_DIR}/"

echo "[install] editable verl..."
pip install -e "${REPO_ROOT}/verl"

echo "[install] smoke checks..."
python -c "import vllm; print('vllm:', vllm.__version__)" 2>/dev/null | tail -1
python -c "from vllm.config import TandemConfig; print('TandemConfig OK')" 2>/dev/null | tail -1
python -c "from vllm.v1.worker.tandem import TandemModelManager; print('TandemModelManager OK')" 2>/dev/null | tail -1
python -c "from vllm.v1.sample.tandem_sampler import TandemSampler; print('TandemSampler OK')" 2>/dev/null | tail -1
python -c "import verl; print('verl:', verl.__file__)"

echo
echo "[install] SUCCESS. Activate with:"
echo "  conda activate ${ENV_NAME}"
