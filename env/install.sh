#!/usr/bin/env bash
# Creates the conda environment described in env/SETUP.md and installs the core
# stack. It stops before the two forks; those are installed by
# third_party/apply_patches.sh, documented in docs/INSTALL.md.
set -euo pipefail

PREFIX="${1:-}"
if [ -z "$PREFIX" ]; then
    echo "usage: bash env/install.sh <conda-env-prefix>" >&2
    echo "  e.g. bash env/install.sh \$HOME/envs/tandem-rlvr" >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 1
fi

conda create -p "$PREFIX" python=3.11 -y

PY="$PREFIX/bin/python"

"$PY" -m pip install --upgrade pip setuptools wheel

# vLLM is installed alone and first: its resolution fixes torch, the CUDA build
# and most of the dependency closure that everything after it has to fit inside.
"$PY" -m pip install "vllm==0.19.1"

# Step above resolves a later transformers than the runs used.
"$PY" -m pip install "transformers==5.5.4"

"$PY" - <<'EOF'
import torch
import transformers
import vllm

print("torch", torch.__version__,
      "| vllm", vllm.__version__,
      "| transformers", transformers.__version__)
EOF

cat <<EOF

Core stack installed at $PREFIX.
Next: docs/INSTALL.md, which builds both forks into this environment.
EOF
