# Training a senior

Install, smoke test, train. Commands run from the repository root and assume the environment
from `docs/INSTALL.md` is on `PATH`.

## 1. Build the two forks

Tandem rollout lives in patched builds of vLLM and verl. `third_party/` holds the pinned upstream
commits and the patches; the script clones and applies them.

```bash
bash third_party/apply_patches.sh
```

Then build and install both, following `docs/INSTALL.md`. Verify:

```bash
python env/smoke.py
```

## 2. Site configuration

```bash
cp train/env.sh.example train/env.sh
# then edit train/env.sh
```

This is the only file holding paths for your machine: the environment's `bin` directory, an
optional `HF_HOME`, the base model, the checkpoint root, the data root, and Weights and Biases
settings. Without a wandb key the launchers use `WANDB_MODE=offline`. It is in `.gitignore`.

## 3. Data

```bash
python data/build_deepscaler.py
```

Writes `data/deepscaler/train.parquet` and `data/deepscaler/heldout.parquet`. The evaluation
parquets ship in `data/eval/`.

## 4. Smoke test

Three steps, and it checks the one thing that fails without an error.

```bash
mkdir -p logs
TOTAL_STEPS=3 SAVE_FREQ=2 TEST_FREQ=3 MAX_RESPONSE=1024 \
  EXP_NAME=tandem_smoke bash train/tandem_grpo.sh 2>&1 | tee logs/smoke.out
grep -o "actor/tandem_senior_token_frac:[^ ]*" logs/smoke.out | tail -1
```

The fraction must be logged and must land near `prob_primary`, 0.5 by default. If it is missing,
the authorship mask is not reaching the loss and the run is plain GRPO. See the first entry under
Known limitations in `README.md`.

## 5. Train

```bash
bash train/tandem_grpo.sh                # tandem rollout, senior only gradient
bash train/vanilla_grpo.sh               # matched control
bash train/kl_reg.sh                     # ablation, KL_COEF required
bash train/grpo_g16.sh                   # ablation, control at 16 rollouts
```

Each is one job on two GPUs. Under a scheduler, request two GPUs while leaving verl at
`trainer.n_gpus_per_node=1`: the second card holds the frozen junior and belongs to the job rather
than to Ray.

Checkpoints land under `${CKPT_ROOT}/${EXP_NAME}`, with Hugging Face weights for each save under
`hf/global_step_N`.

## 6. Pick a checkpoint

Training validates every `TEST_FREQ` steps on `data/deepscaler/heldout.parquet`, a slice
`data/build_deepscaler.py` draws and removes from the training set, and logs
`val-core/macro_avg/acc/pass@4`. Take the step with the highest value.

```bash
SENIOR=${CKPT_ROOT}/${EXP_NAME}/hf/global_step_N
```

Evaluation is in `eval/`; see `eval/README.md`.
