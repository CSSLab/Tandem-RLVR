#!/usr/bin/env bash
# Run the four evaluation phases for one senior checkpoint.
#
#   MODEL=<hub id or path> TAG=<step_120|base> bash eval/run_all.sh
#
# Each phase is a separate process so its vLLM engine is torn down before the
# next one loads, and each phase skips if its JSON already exists, so an
# interrupted sweep is resumed by rerunning the same command.
#
# Knobs, all optional: JUNIOR RESULTS_ROOT BENCHMARKS N LIMIT FRAC GPU_UTIL
# SINGLE_GPU=1. What the phases measure is in eval/README.md.
set -euo pipefail

EVAL_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$EVAL_DIR/.." && pwd)

# train/env.sh is the one site config file: TANDEM_ENV_BIN, HF_HOME, model
# locations. It is optional here, since hub ids work without it.
if [ -f "$ROOT/train/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/train/env.sh"
fi
if [ -n "${TANDEM_ENV_BIN:-}" ]; then
    export PATH="$TANDEM_ENV_BIN:$PATH"
fi
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export TOKENIZERS_PARALLELISM=false

: "${MODEL:?set MODEL to the senior checkpoint}"
: "${TAG:?set TAG to a name for this checkpoint, for example step_120 or base}"
# The junior is the senior's own pre-RL base, by construction, so it defaults to the same
# BASE_MODEL the training launchers initialize from rather than to a second literal that
# would quietly stay behind if BASE_MODEL changed.
JUNIOR=${JUNIOR:-${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/results}
N=${N:-8}
LIMIT=${LIMIT:-0}
FRAC=${FRAC:-0.5}

# Flags are passed only when set, so every default has one definition, in the
# python file that owns it. GPU_UTIL is not forwarded to legibility: scoring
# with prompt_logprobs needs more headroom than generation and has its own
# lower default.
BENCH_ARGS=()
if [ -n "${BENCHMARKS:-}" ]; then BENCH_ARGS=(--benchmarks "$BENCHMARKS"); fi
GPU_ARGS=()
if [ -n "${GPU_UTIL:-}" ]; then GPU_ARGS=(--gpu-util "$GPU_UTIL"); fi
HANDOFF_ARGS=()
if [ -n "${SINGLE_GPU:-}" ]; then HANDOFF_ARGS=(--single-gpu); fi

OUT=$RESULTS_ROOT/$TAG
mkdir -p "$OUT"

phase() {
    local name=$1
    shift
    if [ -f "$OUT/$name.json" ]; then
        echo "skip $name ($TAG)"
        return
    fi
    echo "=== $name ($TAG) ==="
    python "$@"
}

phase solo "$EVAL_DIR/solo.py" \
    --model "$MODEL" --out "$OUT/solo.json" \
    --n "$N" --limit "$LIMIT" "${BENCH_ARGS[@]}" "${GPU_ARGS[@]}"

phase handoff "$EVAL_DIR/handoff.py" \
    --senior "$MODEL" --junior "$JUNIOR" --out "$OUT/handoff.json" \
    --n "$N" --limit "$LIMIT" "${BENCH_ARGS[@]}" "${GPU_ARGS[@]}" "${HANDOFF_ARGS[@]}"


phase legibility "$EVAL_DIR/legibility.py" \
    --solo "$OUT/solo.json" --junior "$JUNIOR" --out "$OUT/legibility.json"

echo "=== done $TAG, results in $OUT ==="
