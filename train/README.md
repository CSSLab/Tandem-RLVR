# Training

Four launchers over one recipe. Each sources `train/env.sh` for site paths and
then calls `python3 -m verl.trainer.main_ppo` with that arm's overrides.

| script | what it trains |
| --- | --- |
| `tandem_grpo.sh` | Treatment arm. Senior and frozen junior co-generate every rollout, the active model is redrawn by a fair coin at word boundaries, the whole response is graded, and the policy gradient is masked to the tokens the senior wrote. Two GPUs, one model each. |
| `vanilla_grpo.sh` | Control arm. The same recipe with the junior removed. Its header enumerates every deliberate difference between the two arms. |
| `kl_reg.sh` | Ablation: solo rollouts plus a per-token KL penalty toward the frozen base, to separate regularisation from co-generation. `KL_COEF` is required. |
| `grpo_g16.sh` | Ablation: `vanilla_grpo.sh` with 16 rollouts per prompt instead of 8. |

`assets/qwen3_word_boundary_ids.json` is the 53,021-id set that defines a word
boundary for Qwen3, and the tandem arm will not start without it.
`assets/build_word_boundary_ids.py` regenerates it from the tokenizer.

## Setup

```
cp train/env.sh.example train/env.sh
```

Then edit it: `TANDEM_ENV_BIN` (the conda environment from `env/install.sh`),
`BASE_MODEL` (senior init, frozen junior, and KL-Reg reference are all this one
model), `CKPT_ROOT`, `DATA_ROOT`, and optionally `HF_HOME` and a wandb key.
Without a key, runs log offline to `./wandb`. `train/env.sh` is not tracked.

Training reads `$DATA_ROOT/deepscaler/train.parquet`, which
`data/build_deepscaler.py` builds. In-training validation reads six of the
parquets shipped under `data/eval/`.

## Launch

Every arm expects two GPUs on one node.

```
bash train/tandem_grpo.sh
bash train/vanilla_grpo.sh
KL_COEF=0.001 bash train/kl_reg.sh
bash train/grpo_g16.sh
```

Every knob in the KNOBS block of a launcher can be overridden from the
environment, and any further argument is passed through to hydra:

```
EXP_NAME=tandem_lr3 LR=3e-6 bash train/tandem_grpo.sh
```

things that matter, a two-GPU allocation and the smoke gate below.

## Confirming the tandem arm is actually running

The trainer logs `actor/tandem_senior_token_frac`, the share of response tokens
the senior wrote. At `PROB_PRIMARY=0.5` it should sit near 0.5. If the metric never
appears at all, the authorship mask never reached the loss: the rollout is
running under stock vLLM instead of the patched build, and the run is plain
GRPO burning a second GPU. A three-step check:

```
TOTAL_STEPS=3 SAVE_FREQ=2 TEST_FREQ=3 EXP_NAME=tandem_smoke bash train/tandem_grpo.sh
```

## Checkpoints

```
$CKPT_ROOT/$EXP_NAME/
    global_step_N/actor/    trainer state for resume, pruned to MAX_CKPT_KEEP
    hf/global_step_N/       HuggingFace weights, one per save, never pruned
```

Resubmitting an unfinished run resumes it from `global_step_N/actor`. The
evaluation reads `hf/global_step_N`, and nothing else under `$CKPT_ROOT`: pass
that path to `eval/run_all.sh`.
