#!/usr/bin/env bash
# Shared activation for the native WSL2 workflow.

set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV_DIR="${PF_VENV:-$REPO_ROOT/.venv}"

pf_activate_require_native_wsl2_ext4() {
  local kernel_release repo_fstype

  if [[ "$(uname -s 2>/dev/null || true)" != "Linux" ]]; then
    echo "[activate_wsl] Native WSL2 is required; Windows and non-Linux hosts are unsupported." >&2
    return 1
  fi

  if [[ -e /.dockerenv || -e /run/.containerenv ]] \
    || grep -Eqi '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
    echo "[activate_wsl] Docker/container execution is unsupported; use native WSL2." >&2
    return 1
  fi

  kernel_release="$(< /proc/sys/kernel/osrelease)" 2>/dev/null || kernel_release=""
  if [[ ! "$kernel_release" =~ [Mm]icrosoft.*WSL2 ]]; then
    echo "[activate_wsl] Native WSL2 is required (WSL1 and generic Linux are unsupported)." >&2
    return 1
  fi

  case "$REPO_ROOT" in
    /mnt/*)
      echo "[activate_wsl] The checkout must be on WSL's Linux filesystem, not under /mnt/." >&2
      return 1
      ;;
  esac

  if ! command -v findmnt >/dev/null 2>&1; then
    echo "[activate_wsl] findmnt is required to verify the checkout filesystem." >&2
    return 1
  fi
  if ! repo_fstype="$(findmnt --noheadings --output FSTYPE --target "$REPO_ROOT" 2>/dev/null)"; then
    echo "[activate_wsl] Could not determine the checkout filesystem for $REPO_ROOT." >&2
    return 1
  fi
  repo_fstype="${repo_fstype//[[:space:]]/}"
  if [[ "$repo_fstype" != "ext4" ]]; then
    echo "[activate_wsl] The checkout must be on native WSL2 ext4 (found: ${repo_fstype:-unknown})." >&2
    return 1
  fi
}

if ! pf_activate_require_native_wsl2_ext4; then
  unset -f pf_activate_require_native_wsl2_ext4
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exit 1
  else
    return 1
  fi
fi
unset -f pf_activate_require_native_wsl2_ext4

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[activate_wsl] Missing native environment: $VENV_DIR" >&2
  echo "[activate_wsl] Create it with Python 3.11 before running evaluations." >&2
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exit 1
  else
    return 1
  fi
fi

# WSL exposes the NVIDIA driver directly. The local toolkit is useful when a
# package (for example llama-cpp-python) needs to compile a CUDA extension.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# Keep model/tokenizer downloads inside the Linux filesystem, not the nearly
# full Windows D: mount. The directory is gitignored.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
