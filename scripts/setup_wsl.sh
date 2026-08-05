#!/usr/bin/env bash
# Idempotent bootstrap for the native WSL2 evaluation and QLoRA environment.

set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() {
  printf '[setup_wsl] %s\n' "$*" >&2
  exit 1
}

require_native_wsl2_ext4() {
  local kernel_release repo_fstype

  [[ "$(uname -s 2>/dev/null || true)" == "Linux" ]] \
    || die "Native WSL2 is required; Windows and non-Linux hosts are unsupported."

  if [[ -e /.dockerenv || -e /run/.containerenv ]] \
    || grep -Eqi '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
    die "Docker/container execution is unsupported; use native WSL2."
  fi

  kernel_release="$(< /proc/sys/kernel/osrelease)" 2>/dev/null || kernel_release=""
  [[ "$kernel_release" =~ [Mm]icrosoft.*WSL2 ]] \
    || die "Native WSL2 is required (WSL1 and generic Linux are unsupported)."

  case "$REPO_ROOT" in
    /mnt/*)
      die "The checkout must be on WSL's Linux filesystem, not under /mnt/."
      ;;
  esac

  command -v findmnt >/dev/null 2>&1 \
    || die "findmnt is required to verify the checkout filesystem."
  if ! repo_fstype="$(findmnt --noheadings --output FSTYPE --target "$REPO_ROOT" 2>/dev/null)"; then
    die "Could not determine the checkout filesystem for $REPO_ROOT."
  fi
  repo_fstype="${repo_fstype//[[:space:]]/}"
  [[ "$repo_fstype" == "ext4" ]] \
    || die "The checkout must be on native WSL2 ext4 (found: ${repo_fstype:-unknown})."
}

# Refuse unsafe execution before creating an environment or installing anything.
require_native_wsl2_ext4
cd "$REPO_ROOT"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || true)}"

if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
  echo "[setup_wsl] uv is required but was not found on PATH." >&2
  exit 1
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "[setup_wsl] Python 3.11 is required but was not found on PATH." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" .venv
fi

# CUDA-specific packages use their own indexes so pip/uv cannot accidentally
# resolve sibling packages from a restricted wheel repository.
"$UV_BIN" pip install --python .venv/bin/python torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$UV_BIN" pip install --python .venv/bin/python llama-cpp-python==0.3.20 \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

"$UV_BIN" pip install --python .venv/bin/python --requirement requirements.txt
"$UV_BIN" pip install --python .venv/bin/python --requirement requirements-training.txt

source "$REPO_ROOT/scripts/activate_wsl.sh"
python scripts/verify_gpu.py

echo "[setup_wsl] Native WSL2 environment is ready."
