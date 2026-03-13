# vLLM-Native Tandem Training with GRPO

Tandem Training (TT) for LLMs via native vLLM integration. A hot (primary) model and a frozen model generate tokens together during rollout — a per-token bernoulli selector picks which model's token to use. The hot model trains via GRPO on the mixed sequence, with loss weighted by which model produced each token.

Built on vLLM v0.8.5 (v1 engine) + verl 0.5.0.

## Performance

| Setup | Step time | Overhead |
|-------|-----------|----------|
| Vanilla GRPO | 18.6s | 1.0x |
| **vLLM-native TT** | **33.5s** | **1.8x** |
| Old HF-loop TT | ~550s | ~30x |

Qwen3-0.6B, 2x A100, GSM8K, B=8, N=8, L=512.

## Setup

```bash
# 1. Clone
git clone https://github.com/CSSLab/llm-tandem-verl.git
cd llm-tandem-verl

# 2. Create conda env
conda env create -f environment.yml

# 3. Symlink vllm_source into the env (this is how tandem edits reach vLLM)
ENVDIR=$(conda info --envs | grep tandem-verl | awk '{print $NF}')
rm -rf $ENVDIR/lib/python3.10/site-packages/vllm
ln -s $(pwd)/vllm_source/vllm $ENVDIR/lib/python3.10/site-packages/vllm

# 4. Install verl from the repo (editable)
conda activate tandem-verl
cd verl && pip install -e . && cd ..
```

## Run

```bash
# Tandem GRPO training (GPU 0: hot model + training, GPU 1: frozen model)
cd verl
bash run_tandem_native_grpo_gsm8k.sh

# Vanilla GRPO benchmark (both GPUs for standard training)
bash run_vanilla_grpo_gsm8k_benchmark.sh
```

Training scripts expect GSM8K parquet files at `scratch/data/gsm8k/{train,test}.parquet` and model weights cached under `scratch/models/`. Edit `SCRATCH_DIR` and `MODEL_NAME` in the scripts as needed.

## Key metrics to watch

- `tandem/primary_token_fraction` — fraction of tokens from hot model (~0.5 with bernoulli p=0.5)
- `tandem/frozen_token_fraction` — fraction from frozen model
- `actor/pg_loss` — policy gradient loss (should be lower than vanilla due to weighted masking)

## Architecture

```
GPU 0: vLLM primary model (rollout) + FSDP training (actor update)
GPU 1: vLLM frozen model (rollout only, no gradients)
```

During generation, both models run forward passes. A `TandemSampler` selects tokens per-position. The resulting `model_mask` flows through the vLLM engine pipeline back to verl's actor, where frozen-token positions receive reduced loss weight (`tandem_jr_tkn_weight`, default 0.2).

Edits to vLLM source are in `vllm_source/vllm/v1/` (worker, sampler, config, outputs, scheduler, engine). Edits to verl are in `verl/verl/workers/` (actor, rollout).
