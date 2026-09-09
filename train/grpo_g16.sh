#!/usr/bin/env bash
# ============================================================================
# Ablation: vanilla GRPO with twice the rollouts per prompt.
# Tandem generation runs two models over every sequence, so this arm answers
# what the control does with a comparable rollout budget. ROLLOUT_N is the only
# value that changes; everything else comes from train/vanilla_grpo.sh, which
# also sources train/env.sh.
#   Launch:  bash train/grpo_g16.sh
# ============================================================================
set -euo pipefail
export ROLLOUT_N=16
export EXP_NAME=${EXP_NAME:-vanilla_grpo_qwen3_4b_deepscaler_g16}
exec bash "$(cd "$(dirname "$0")" && pwd)/vanilla_grpo.sh" "$@"
