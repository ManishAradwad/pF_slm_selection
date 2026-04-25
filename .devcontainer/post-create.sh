#!/usr/bin/env bash
# Runs once on container creation (devcontainer postCreateCommand).
# Idempotent — safe to re-run by hand if you ever need to.

set -euo pipefail

# DYNAMIC PATH: Grabs the current working directory automatically
REPO=$(pwd)

# 1. Claude Code CLI (Installed globally via NPM to bypass PATH issues)
if ! command -v claude &>/dev/null; then
  echo "[post-create] Installing Claude Code CLI globally..."
  npm install -g @anthropic-ai/claude-code
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

# 3. GPU sanity check. 
"$VENV/bin/python" "$REPO/scripts/verify_gpu.py"