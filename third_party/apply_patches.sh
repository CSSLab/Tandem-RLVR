#!/usr/bin/env bash
# Reconstruct both forks: clone the upstreams at their pinned commits into
# third_party/{vllm,verl} and apply the tandem patches.
#
#   bash third_party/apply_patches.sh              clone and patch
#   bash third_party/apply_patches.sh --install    then install both, editable
#
# Overrides: PYTHON (default: python), VLLM_SRC, VERL_SRC.
#
# Rerunning is safe. A tree that already carries its patch is left untouched, and a
# tree that is dirty or sits on an unexpected commit is reported rather than repaired,
# because the alternative is throwing away someone's work in place.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PYTHON=${PYTHON:-python}
VLLM_SRC=${VLLM_SRC:-$HERE/vllm}
VERL_SRC=${VERL_SRC:-$HERE/verl}
BRANCH=tandem-rlvr

INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=1 ;;
        *) echo "usage: $0 [--install]" >&2; exit 2 ;;
    esac
done

log() { echo "[apply_patches] $*"; }
die() { echo "[apply_patches] error: $*" >&2; exit 1; }

command -v git >/dev/null || die "git not found on PATH"

# clone_at <url> <dest> <commit>
clone_at() {
    local url=$1 dest=$2 commit=$3
    if [ -e "$dest" ]; then
        [ -d "$dest/.git" ] || die "$dest exists and is not a git checkout"
        return
    fi
    local tmp="$dest.incoming.$$"
    rm -rf "$tmp"
    log "cloning $url into $dest"
    # Blobless: the pinned commits are reachable from the default branch, and the full
    # vLLM object store is about a gigabyte of history nothing here reads.
    if ! git clone --filter=blob:none "$url" "$tmp" 2>/dev/null; then
        rm -rf "$tmp"
        git clone "$url" "$tmp" || { rm -rf "$tmp"; die "clone of $url failed"; }
    fi
    mv "$tmp" "$dest"
    git -C "$dest" cat-file -e "${commit}^{commit}" 2>/dev/null \
        || die "$dest does not contain pinned commit $commit"
}

# patch_fork <name> <dest> <commit> <patch>
patch_fork() {
    local name=$1 dest=$2 commit=$3 patch=$4
    [ -f "$patch" ] || die "patch file missing: $patch"

    if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then
        log "$name: patch already applied"
        return
    fi
    if [ -n "$(git -C "$dest" status --porcelain)" ]; then
        die "$name: working tree at $dest has local changes, refusing to patch it"
    fi
    log "$name: checking out $commit on branch $BRANCH"
    git -C "$dest" checkout -B "$BRANCH" "$commit" >/dev/null 2>&1 \
        || die "$name: could not check out $commit in $dest"
    git -C "$dest" apply --check "$patch" \
        || die "$name: $(basename "$patch") does not apply at $commit. The patches are pinned to that commit and are not maintained against other bases."
    git -C "$dest" apply "$patch"
    git -C "$dest" add -A
    git -C "$dest" -c user.name="tandem-rlvr" -c user.email="tandem-rlvr@localhost" \
        commit -q -m "tandem edits ($name)"
    log "$name: patch applied and committed on $BRANCH"
}

VLLM_COMMIT=$(head -1 "$HERE/VLLM_BASE_COMMIT.txt" | tr -d '[:space:]')
VERL_COMMIT=$(head -1 "$HERE/VERL_BASE_COMMIT.txt" | tr -d '[:space:]')
[ -n "$VLLM_COMMIT" ] || die "VLLM_BASE_COMMIT.txt is empty"
[ -n "$VERL_COMMIT" ] || die "VERL_BASE_COMMIT.txt is empty"

clone_at https://github.com/vllm-project/vllm.git "$VLLM_SRC" "$VLLM_COMMIT"
patch_fork vllm "$VLLM_SRC" "$VLLM_COMMIT" "$HERE/vllm-tandem.patch"

clone_at https://github.com/volcengine/verl.git "$VERL_SRC" "$VERL_COMMIT"
patch_fork verl "$VERL_SRC" "$VERL_COMMIT" "$HERE/verl-tandem.patch"

if [ "$INSTALL" = "0" ]; then
    log "done. Install with --install, or follow docs/INSTALL.md."
    exit 0
fi

command -v "$PYTHON" >/dev/null || die "PYTHON=$PYTHON not found"
log "installing vllm from $VLLM_SRC"
# The tandem edits are pure Python. VLLM_USE_PRECOMPILED reuses the release wheel's
# CUDA extensions instead of building them, which needs no toolchain and takes seconds.
VLLM_USE_PRECOMPILED=1 "$PYTHON" -m pip install -e "$VLLM_SRC" --no-build-isolation \
    || die "vllm install failed. See the section on compiled extensions in docs/INSTALL.md."

log "installing verl from $VERL_SRC"
# Base install, not the [vllm] extra: that extra pins vllm<=0.12.0 and would replace
# the fork just installed.
"$PYTHON" -m pip install -e "$VERL_SRC" -c "$ROOT/env/constraints.txt" || die "verl install failed"
"$PYTHON" -m pip install "TransferQueue==0.1.8" || die "TransferQueue install failed"

log "verifying"
"$PYTHON" - <<'PY' || die "the installed vllm has no tandem support. Check that pip picked up third_party/vllm and not the release wheel."
from vllm.config import TandemConfig
from vllm.v1.sample.tandem_sampler import TandemSampler
print("[apply_patches] vllm fork OK")
PY
"$PYTHON" - <<'PY' || die "the installed verl has no tandem gate"
import transfer_queue  # noqa: F401
from verl.workers.utils.losses import _apply_tandem_senior_gate  # noqa: F401
print("[apply_patches] verl fork OK")
PY
log "done"
