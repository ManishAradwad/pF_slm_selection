#!/usr/bin/env bash
# Clone, pin, and build the official llama.cpp LFM2 toolchain in native WSL.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly LOCK_FILE="$REPO_ROOT/configs/lfm25/llama_cpp.lock.json"
readonly UPSTREAM_ROOT="$REPO_ROOT/UPSTREAM"
readonly CHECKOUT="$UPSTREAM_ROOT/llama.cpp"
readonly BUILD_DIR="$CHECKOUT/build-lfm25-cuda"

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/setup_lfm25_llama_cpp.sh [--offline] [--allow-portable-binaries]

Clone/fetch only the official ggml-org/llama.cpp repository, check out the
revision in configs/lfm25/llama_cpp.lock.json, and build CUDA-enabled
llama-cli, llama-quantize, and llama-bench under ignored UPSTREAM/llama.cpp.

Options:
  --offline                  Refuse network access; require the pinned commit
                             to exist locally.
  --allow-portable-binaries Accept binary SHA-256 differences caused by a
                             different compiler/toolchain after all pinned
                             source and runtime checks pass. The exact hashes
                             in the lock are required by default.
  -h, --help                 Show this help.

Environment:
  CUDA_HOME                         Linux CUDA toolkit (default: /usr/local/cuda)
  LLAMA_CPP_CUDA_ARCHITECTURES      CMake CUDA architecture (default: lock value)
  LLAMA_CPP_JOBS                    Parallel build jobs (default: min(nproc, 8))
EOF
}

offline=0
allow_portable_binaries=0
while (($#)); do
  case "$1" in
    --offline) offline=1 ;;
    --allow-portable-binaries) allow_portable_binaries=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done

[[ "$(uname -s)" == Linux ]] || die 'Run this script inside native WSL2, not Windows.'
grep -qi microsoft /proc/sys/kernel/osrelease \
  || die 'Run this script inside native WSL2.'
[[ ! -e /.dockerenv ]] || die 'Docker is not supported; use the native WSL2 checkout.'
case "$REPO_ROOT" in
  /mnt/*) die "Move the checkout to the WSL Linux filesystem (for example, /home/<user>/...)." ;;
esac

[[ -f "$LOCK_FILE" ]] || die "Missing lock file: $LOCK_FILE"
for command_name in git python3 cmake ninja g++ sha256sum flock; do
  command -v "$command_name" >/dev/null 2>&1 \
    || die "Required command is unavailable: $command_name"
done

lock_get() {
  python3 - "$LOCK_FILE" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for component in sys.argv[2].split("."):
    value = value[component]
if isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(",", ":")))
else:
    print(value)
PY
}

OFFICIAL_URL="$(lock_get upstream.repository)"
readonly OFFICIAL_URL
PINNED_COMMIT="$(lock_get upstream.commit)"
readonly PINNED_COMMIT
PINNED_TREE="$(lock_get upstream.tree)"
readonly PINNED_TREE
LOCKED_CUDA_ARCH="$(lock_get build.options.CMAKE_CUDA_ARCHITECTURES)"
readonly LOCKED_CUDA_ARCH
readonly EXPECTED_URL='https://github.com/ggml-org/llama.cpp.git'

[[ "$OFFICIAL_URL" == "$EXPECTED_URL" ]] \
  || die "Lock file repository is not the approved official upstream: $OFFICIAL_URL"
[[ "$PINNED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'Lock file commit is not a full Git SHA.'
[[ "$PINNED_TREE" =~ ^[0-9a-f]{40}$ ]] || die 'Lock file tree is not a full Git SHA.'

mkdir -p "$UPSTREAM_ROOT"
if [[ ! -e "$CHECKOUT" ]]; then
  ((offline == 0)) || die "Offline mode: checkout is missing at $CHECKOUT"
  git clone --filter=blob:none --no-tags "$OFFICIAL_URL" "$CHECKOUT"
elif [[ ! -d "$CHECKOUT/.git" ]]; then
  die "Refusing to replace non-Git path: $CHECKOUT"
fi

actual_url="$(git -C "$CHECKOUT" remote get-url origin)"
[[ "$actual_url" == "$OFFICIAL_URL" ]] \
  || die "Unexpected origin URL in $CHECKOUT: $actual_url"

# Fetch is the only network operation in this script. Disable pushes even if a
# user accidentally runs `git push` from the ignored upstream checkout.
git -C "$CHECKOUT" remote set-url --push origin DISABLED

if ! git -C "$CHECKOUT" cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null; then
  ((offline == 0)) || die "Offline mode: pinned commit is not available locally."
  git -C "$CHECKOUT" fetch --filter=blob:none --no-tags origin "$PINNED_COMMIT"
fi

if ! git -C "$CHECKOUT" diff --quiet || ! git -C "$CHECKOUT" diff --cached --quiet; then
  die 'Tracked changes exist in the upstream checkout; refusing to switch revisions.'
fi
git -C "$CHECKOUT" checkout --detach "$PINNED_COMMIT"
[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$PINNED_COMMIT" ]] \
  || die 'Pinned checkout verification failed.'
[[ "$(git -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$PINNED_TREE" ]] \
  || die 'Pinned checkout tree does not match the lock file.'

python3 - "$LOCK_FILE" "$CHECKOUT" <<'PY'
import hashlib
import json
import pathlib
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
checkout = pathlib.Path(sys.argv[2])
for relative, expected in lock["source_sha256"].items():
    path = checkout / relative
    if not path.is_file():
        raise SystemExit(f"missing pinned source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"pinned source hash mismatch: {relative}")
print(f"[setup_lfm25_llama_cpp.sh] Verified {len(lock['source_sha256'])} pinned source hashes.")
PY

grep -Fq '@ModelBase.register("Lfm2ForCausalLM", "LFM2ForCausalLM")' \
  "$CHECKOUT/conversion/lfm2.py" || die 'Pinned converter lacks LFM2 registration.'
grep -Fq 'LLM_ARCH_LFM2' "$CHECKOUT/src/llama-arch.h" \
  || die 'Pinned runtime lacks the LFM2 architecture.'

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
[[ -x "$cuda_home/bin/nvcc" ]] || die "Linux nvcc is missing: $cuda_home/bin/nvcc"
[[ -f "$cuda_home/lib64/libcudart.so" ]] || die "Linux libcudart is missing under $cuda_home/lib64"
cuda_arch="${LLAMA_CPP_CUDA_ARCHITECTURES:-$LOCKED_CUDA_ARCH}"
[[ "$cuda_arch" =~ ^[0-9]+([;-][0-9]+)*$ ]] \
  || die "Invalid LLAMA_CPP_CUDA_ARCHITECTURES: $cuda_arch"

if [[ -n "${LLAMA_CPP_JOBS:-}" ]]; then
  jobs="$LLAMA_CPP_JOBS"
else
  jobs="$(nproc)"
  ((jobs > 8)) && jobs=8
fi
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "Invalid LLAMA_CPP_JOBS: $jobs"

# Prevent two setup processes from racing Ninja in the same build tree.
exec 9>"$CHECKOUT/.lfm25-build.lock"
flock 9

# Exclude Windows CUDA executables inherited through WSL's normal PATH.
build_path="$cuda_home/bin:/usr/local/bin:/usr/bin:/bin"
env PATH="$build_path" CUDA_HOME="$cuda_home" CUDACXX="$cuda_home/bin/nvcc" \
  cmake -S "$CHECKOUT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$cuda_home/bin/nvcc" \
    -DCUDAToolkit_ROOT="$cuda_home" \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NCCL=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_LLGUIDANCE=OFF

env PATH="$build_path" CUDA_HOME="$cuda_home" \
  cmake --build "$BUILD_DIR" --target llama-cli llama-quantize llama-bench \
    --parallel "$jobs"

readonly CLI="$BUILD_DIR/bin/llama-cli"
readonly QUANTIZE="$BUILD_DIR/bin/llama-quantize"
readonly BENCH="$BUILD_DIR/bin/llama-bench"
[[ -x "$CLI" ]] || die "Build did not produce $CLI"
[[ -x "$QUANTIZE" ]] || die "Build did not produce $QUANTIZE"
[[ -x "$BENCH" ]] || die "Build did not produce $BENCH"

python3 - "$LOCK_FILE" "$BUILD_DIR/bin" "$allow_portable_binaries" <<'PY'
import hashlib
import json
import pathlib
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
binary_dir = pathlib.Path(sys.argv[2])
allow_portable = sys.argv[3] == "1"
required = ("llama-cli", "llama-quantize", "llama-bench")
expected_hashes = lock["build"]["binary_sha256_observed"]
if set(expected_hashes) != set(required):
    raise SystemExit("lock must contain hashes for llama-cli, llama-quantize, and llama-bench")

mismatches = []
for name in required:
    path = binary_dir / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = expected_hashes[name]
    if actual != expected:
        mismatches.append((name, expected, actual))

if mismatches and not allow_portable:
    details = "; ".join(
        f"{name}: expected {expected}, got {actual}"
        for name, expected, actual in mismatches
    )
    raise SystemExit(
        "built binary hash mismatch (use --allow-portable-binaries only for an "
        f"intentional compiler/toolchain variance): {details}"
    )
if mismatches:
    for name, expected, actual in mismatches:
        print(
            f"[setup_lfm25_llama_cpp.sh] WARNING: explicitly accepting portable "
            f"{name} hash; lock={expected}, actual={actual}",
            file=sys.stderr,
        )
else:
    print("[setup_lfm25_llama_cpp.sh] Verified all locked binary SHA-256 hashes.")
PY

python3 - "$LOCK_FILE" "$BUILD_DIR/CMakeCache.txt" "$cuda_arch" <<'PY'
import json
import pathlib
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
cache = {}
for line in pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
    if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
        continue
    key_type, value = line.split("=", 1)
    key, _ = key_type.split(":", 1)
    cache[key] = value

required = dict(lock["build"]["options"])
required["CMAKE_CUDA_ARCHITECTURES"] = sys.argv[3]
for key, expected in required.items():
    actual = cache.get(key)
    if actual != str(expected):
        raise SystemExit(f"CMake cache mismatch for {key}: expected {expected}, got {actual}")
print("[setup_lfm25_llama_cpp.sh] Verified pinned CMake options.")
PY

version_output="$("$CLI" --version 2>&1)"
grep -Fq "${PINNED_COMMIT:0:8}" <<<"$version_output" \
  || die 'llama-cli version does not match the pinned commit.'
device_output="$("$CLI" --list-devices 2>&1)"
grep -Fq 'CUDA0:' <<<"$device_output" || die 'CUDA device was not registered by llama-cli.'
# llama-quantize intentionally exits nonzero after displaying help.
quantize_help="$("$QUANTIZE" --help 2>&1 || true)"
grep -Fq 'Q4_K_M' <<<"$quantize_help" \
  || die 'llama-quantize does not advertise Q4_K_M support.'
bench_help="$("$BENCH" --help 2>&1)"
grep -Fq -- '--n-prompt' <<<"$bench_help" \
  || die 'llama-bench does not advertise prompt-processing benchmarks.'

printf '[%s] Ready at pinned commit %s.\n' "$SCRIPT_NAME" "$PINNED_COMMIT"
printf '[%s] CUDA device: %s\n' "$SCRIPT_NAME" "$(grep -F 'CUDA0:' <<<"$device_output" | sed 's/^[[:space:]]*//')"
printf '[%s] Binaries: %s, %s, and %s\n' "$SCRIPT_NAME" "$CLI" "$QUANTIZE" "$BENCH"
