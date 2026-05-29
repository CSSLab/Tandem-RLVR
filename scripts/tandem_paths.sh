# Shared path exports for Tandem-RLVR launch scripts.
# Usage: source "$(dirname "$0")/../scripts/tandem_paths.sh"
TANDEM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TANDEM_REPO_ROOT
export TANDEM_SCRATCH="${TANDEM_SCRATCH:-${TANDEM_REPO_ROOT}/scratch}"
export HF_HOME="${HF_HOME:-${TANDEM_SCRATCH}/models}"
