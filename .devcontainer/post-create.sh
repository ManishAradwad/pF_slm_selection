#!/usr/bin/env bash
# Runs once on container creation (devcontainer postCreateCommand).
# Idempotent — safe to re-run by hand if you ever need to.

set -euo pipefail

REPO=/workspaces/pF_slm_selection

# 1. Claude Code CLI (was the prior postCreateCommand's only job).
if ! command -v claude &>/dev/null; then
  curl -fsSL https://claude.ai/install.sh | bash
fi

# 2. (Re)create the pf_docker shim venv.
# --system-site-packages: thin shim. Heavy deps live in the image's system
# Python (built by the Dockerfile); the venv is just the familiar
# `source pf_docker/bin/activate` entry point and a place to install ad-hoc
# experimental packages without polluting system Python. This makes rebuilds
# fast even though the venv directory itself survives via the workspace
# bind mount — a stale interpreter pointer (the 2026-04-24 failure mode)
# is detected and rebuilt here.
VENV="$REPO/pf_docker"
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys' &>/dev/null; then
  echo "[post-create] (re)creating $VENV"
  rm -rf "$VENV"
  python3.11 -m venv --system-site-packages "$VENV"
else
  echo "[post-create] $VENV looks healthy — leaving as-is"
fi

# 3. GPU sanity check. Fails loudly if llama-cpp-python lost CUDA — better
# than every eval silently falling back to CPU.
"$VENV/bin/python" "$REPO/scripts/verify_gpu.py"
