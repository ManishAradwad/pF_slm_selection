#!/usr/bin/env bash
# Convert a local merged LFM2.5 HF checkpoint and create deployable GGUFs.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly LOCK_FILE="$REPO_ROOT/configs/lfm25/llama_cpp.lock.json"
readonly CHECKOUT="$REPO_ROOT/UPSTREAM/llama.cpp"
readonly BUILD_DIR="$CHECKOUT/build-lfm25-cuda"
readonly CONVERTER="$CHECKOUT/convert_hf_to_gguf.py"
readonly CLI="$BUILD_DIR/bin/llama-cli"
readonly QUANTIZE="$BUILD_DIR/bin/llama-quantize"
readonly BENCH="$BUILD_DIR/bin/llama-bench"

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/convert_lfm25_gguf.sh [OPTIONS] MERGED_HF_DIR OUTPUT_PREFIX

Convert a local merged LFM2.5 Hugging Face directory into a 16-bit reference,
Q8_0, and Q4_K_M. OUTPUT_PREFIX must not include the .gguf extension.

Options:
  --reference bf16|f16  Reference tensor type (default: bf16).
  --q5                  Also create Q5_K_M.
  --allow-portable-binaries  Explicitly accept toolchain-dependent binary hashes.
  -h, --help            Show this help.

The script never downloads, uploads, deletes, or overwrites model artifacts.
An interrupted run leaves its uniquely named partial file for inspection.
EOF
}

reference_type=bf16
make_q5=0
allow_portable_binaries=0
positionals=()
while (($#)); do
  case "$1" in
    --reference)
      (($# >= 2)) || die '--reference requires bf16 or f16.'
      reference_type="${2,,}"
      shift 2
      ;;
    --q5) make_q5=1; shift ;;
    --allow-portable-binaries) allow_portable_binaries=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; positionals+=("$@"); break ;;
    -*) die "Unknown option: $1" ;;
    *) positionals+=("$1"); shift ;;
  esac
done

[[ "$reference_type" == bf16 || "$reference_type" == f16 ]] \
  || die '--reference must be bf16 or f16.'
((${#positionals[@]} == 2)) || { usage >&2; exit 2; }

[[ "$(uname -s)" == Linux ]] || die 'Run this script inside native WSL2, not Windows.'
grep -qi microsoft /proc/sys/kernel/osrelease || die 'Run this script inside native WSL2.'
[[ ! -e /.dockerenv ]] || die 'Docker is not supported; use native WSL2.'
case "$REPO_ROOT" in
  /mnt/*) die 'The repository must be on the WSL Linux filesystem.' ;;
esac

for command_name in python3 realpath sha256sum flock timeout; do
  command -v "$command_name" >/dev/null 2>&1 \
    || die "Required command is unavailable: $command_name"
done
[[ -f "$LOCK_FILE" ]] || die "Missing lock file: $LOCK_FILE"
[[ -d "$CHECKOUT/.git" ]] || die 'Pinned llama.cpp checkout is missing; run setup_lfm25_llama_cpp.sh.'
[[ -x "$CLI" && -x "$QUANTIZE" && -x "$BENCH" && -f "$CONVERTER" ]] \
  || die 'Pinned llama.cpp tools are missing; run setup_lfm25_llama_cpp.sh.'

lock_get() {
  python3 - "$LOCK_FILE" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for component in sys.argv[2].split("."):
    value = value[component]
print(value)
PY
}

PINNED_COMMIT="$(lock_get upstream.commit)"
readonly PINNED_COMMIT
PINNED_TREE="$(lock_get upstream.tree)"
readonly PINNED_TREE
OFFICIAL_URL="$(lock_get upstream.repository)"
readonly OFFICIAL_URL
[[ "$OFFICIAL_URL" == 'https://github.com/ggml-org/llama.cpp.git' ]] \
  || die 'Lock file does not name the approved official llama.cpp repository.'
[[ "$PINNED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'Lock commit is not a full Git SHA.'
[[ "$PINNED_TREE" =~ ^[0-9a-f]{40}$ ]] || die 'Lock tree is not a full Git SHA.'
[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$PINNED_COMMIT" ]] \
  || die 'llama.cpp checkout is not at the pinned commit; rerun setup.'
[[ "$(git -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$PINNED_TREE" ]] \
  || die 'llama.cpp checkout tree does not match the lock; rerun setup.'
[[ "$(git -C "$CHECKOUT" remote get-url origin)" == "$OFFICIAL_URL" ]] \
  || die 'llama.cpp origin is not the approved official repository.'
if ! git -C "$CHECKOUT" diff --quiet || ! git -C "$CHECKOUT" diff --cached --quiet; then
  die 'Tracked llama.cpp source changes exist; refusing conversion.'
fi

python3 - "$LOCK_FILE" "$CHECKOUT" "$BUILD_DIR/bin" "$allow_portable_binaries" <<'PY'
import hashlib
import json
import pathlib
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
checkout = pathlib.Path(sys.argv[2])
binary_dir = pathlib.Path(sys.argv[3])
allow_portable = sys.argv[4] == "1"
for relative, expected in lock["source_sha256"].items():
    path = checkout / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"pinned llama.cpp source mismatch: {relative}")

required = ("llama-cli", "llama-quantize", "llama-bench")
expected_hashes = lock["build"]["binary_sha256_observed"]
if set(expected_hashes) != set(required):
    raise SystemExit("lock must contain hashes for llama-cli, llama-quantize, and llama-bench")
mismatches = []
for name in required:
    actual = hashlib.sha256((binary_dir / name).read_bytes()).hexdigest()
    expected = expected_hashes[name]
    if actual != expected:
        mismatches.append((name, expected, actual))
if mismatches and not allow_portable:
    details = "; ".join(
        f"{name}: expected {expected}, got {actual}"
        for name, expected, actual in mismatches
    )
    raise SystemExit(
        "llama.cpp binary hash mismatch (rerun setup, or use "
        f"--allow-portable-binaries for intentional toolchain variance): {details}"
    )
if mismatches:
    for name, expected, actual in mismatches:
        print(
            f"[convert_lfm25_gguf.sh] WARNING: explicitly accepting portable "
            f"{name} hash; lock={expected}, actual={actual}",
            file=sys.stderr,
        )
else:
    print("[convert_lfm25_gguf.sh] Verified all locked binary SHA-256 hashes.")
PY

version_output="$("$CLI" --version 2>&1)"
grep -Fq "${PINNED_COMMIT:0:8}" <<<"$version_output" \
  || die 'llama-cli runtime does not identify the pinned commit.'
quantize_help="$("$QUANTIZE" --help 2>&1 || true)"
grep -Fq 'Q4_K_M' <<<"$quantize_help" \
  || die 'llama-quantize does not advertise Q4_K_M support.'
bench_help="$("$BENCH" --help 2>&1)"
grep -Fq -- '--n-prompt' <<<"$bench_help" \
  || die 'llama-bench does not advertise prompt-processing benchmarks.'

input_arg="${positionals[0]}"
output_arg="${positionals[1]}"
case "$input_arg$output_arg" in
  *://*) die 'Only local filesystem input and output paths are accepted.' ;;
esac
[[ -d "$input_arg" ]] || die "Merged HF directory does not exist: $input_arg"
INPUT_DIR="$(realpath -e "$input_arg")"
readonly INPUT_DIR
case "$INPUT_DIR" in
  /mnt/*) die 'Merged weights must remain on the WSL Linux filesystem.' ;;
esac

[[ "$output_arg" != *.gguf ]] || die 'OUTPUT_PREFIX must not include .gguf.'
output_parent="$(dirname "$output_arg")"
mkdir -p "$output_parent"
OUTPUT_PREFIX="$(realpath -m "$output_arg")"
readonly OUTPUT_PREFIX
case "$OUTPUT_PREFIX" in
  /mnt/*) die 'GGUF artifacts must remain on the WSL Linux filesystem.' ;;
esac

readonly PYTHON_BIN="${LLAMA_CPP_CONVERTER_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || die "Converter Python is unavailable: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import numpy, safetensors, sentencepiece, torch, transformers' \
  || die 'Converter Python is missing a required local package.'

"$PYTHON_BIN" - "$INPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
config_path = root / "config.json"
if not config_path.is_file():
    raise SystemExit("merged HF directory is missing config.json")
config = json.loads(config_path.read_text(encoding="utf-8"))
architectures = config.get("architectures") or []
accepted = {"Lfm2ForCausalLM", "LFM2ForCausalLM"}
if config.get("model_type") != "lfm2" or not accepted.intersection(architectures):
    raise SystemExit("merged HF directory is not LFM2/Lfm2ForCausalLM")
if not list(root.glob("*.safetensors")):
    raise SystemExit("merged HF directory has no safetensors weights")
if not (root / "tokenizer.json").is_file():
    raise SystemExit("merged HF directory is missing tokenizer.json")
print("[convert_lfm25_gguf.sh] Input contract: lfm2 / Lfm2ForCausalLM")
PY

readonly REF_LABEL="${reference_type^^}"
readonly REFERENCE_GGUF="${OUTPUT_PREFIX}-${REF_LABEL}.gguf"
readonly Q8_GGUF="${OUTPUT_PREFIX}-Q8_0.gguf"
readonly Q4_GGUF="${OUTPUT_PREFIX}-Q4_K_M.gguf"
readonly Q5_GGUF="${OUTPUT_PREFIX}-Q5_K_M.gguf"
readonly MANIFEST="${OUTPUT_PREFIX}-conversion.json"

# Serialize all reads and writes sharing one output prefix. The lock remains as
# local audit evidence and is harmless on later idempotent runs.
exec 8>"${OUTPUT_PREFIX}.conversion.lock"
flock 8

if [[ -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" - \
    "$MANIFEST" "$INPUT_DIR" "$OUTPUT_PREFIX" "$PINNED_COMMIT" "$PINNED_TREE" \
    "$reference_type" "$REFERENCE_GGUF" "$Q8_GGUF" "$Q4_GGUF" "$Q5_GGUF" \
    "$make_q5" "$BUILD_DIR/bin" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    manifest_path,
    input_directory,
    output_prefix,
    pinned_commit,
    pinned_tree,
    reference_type,
    reference_path,
    q8_path,
    q4_path,
    q5_path,
    make_q5,
    binary_directory,
) = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
input_dir = pathlib.Path(input_directory)

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

inputs = {
    path.name: sha256_file(path)
    for path in sorted(input_dir.iterdir())
    if path.is_file() and path.suffix in {".json", ".jinja", ".model", ".safetensors"}
}
if manifest.get("input_config_sha256") != inputs["config.json"]:
    raise SystemExit("existing conversion manifest belongs to different input config")
locked_inputs = manifest.get("input_files_sha256")
if locked_inputs is None:
    raise SystemExit("existing conversion manifest lacks merged HF fingerprints")
if locked_inputs != inputs:
    raise SystemExit("existing conversion manifest belongs to different merged HF files")
if manifest.get("schema_version") != 2:
    raise SystemExit(
        "existing conversion manifest is not hash-bearing schema version 2; "
        "refusing unverified artifact reuse"
    )
if manifest.get("output_prefix_name") != pathlib.Path(output_prefix).name:
    raise SystemExit("existing conversion manifest belongs to a different output prefix")
if manifest.get("llama_cpp_commit") != pinned_commit:
    raise SystemExit("existing conversion manifest used a different llama.cpp commit")
if manifest.get("llama_cpp_tree") != pinned_tree:
    raise SystemExit("existing conversion manifest used a different llama.cpp tree")

binary_dir = pathlib.Path(binary_directory)
current_binary_hashes = {
    name: sha256_file(binary_dir / name)
    for name in ("llama-cli", "llama-quantize", "llama-bench")
}
if manifest.get("llama_cpp_binary_sha256") != current_binary_hashes:
    raise SystemExit("existing conversion manifest used different llama.cpp binaries")

reference_label = reference_type.upper()
expected_paths = {
    reference_label: pathlib.Path(reference_path),
    "Q8_0": pathlib.Path(q8_path),
    "Q4_K_M": pathlib.Path(q4_path),
    "Q5_K_M": pathlib.Path(q5_path),
}
artifacts = manifest.get("artifacts")
if not isinstance(artifacts, dict):
    raise SystemExit("existing conversion manifest has malformed artifacts")
required_labels = {reference_label, "Q8_0", "Q4_K_M"}
if make_q5 == "1" or expected_paths["Q5_K_M"].exists():
    required_labels.add("Q5_K_M")
if not required_labels.issubset(artifacts):
    missing = sorted(required_labels.difference(artifacts))
    raise SystemExit(f"existing conversion manifest lacks required artifacts: {missing}")

for label, record in artifacts.items():
    if label not in expected_paths:
        raise SystemExit(f"existing conversion manifest has unexpected artifact: {label}")
    if not isinstance(record, dict):
        raise SystemExit(f"existing conversion manifest has malformed artifact: {label}")
    path = expected_paths[label]
    if record.get("file") != path.name:
        raise SystemExit(f"existing conversion manifest filename mismatch: {label}")
    if not path.is_file():
        raise SystemExit(f"manifested artifact is missing: {path.name}")
    expected_size = record.get("bytes")
    expected_hash = record.get("sha256")
    if not isinstance(expected_size, int) or expected_size <= 4:
        raise SystemExit(f"invalid manifested artifact size: {label}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise SystemExit(f"missing or invalid manifested artifact SHA-256: {label}")
    try:
        int(expected_hash, 16)
    except ValueError as error:
        raise SystemExit(f"invalid manifested artifact SHA-256: {label}") from error
    if path.stat().st_size != expected_size:
        raise SystemExit(f"manifested artifact size mismatch: {label}")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise SystemExit(f"artifact does not have GGUF magic: {path.name}")
    if sha256_file(path) != expected_hash:
        raise SystemExit(f"manifested artifact SHA-256 mismatch: {label}")
print(f"[convert_lfm25_gguf.sh] Verified {len(artifacts)} manifested GGUF artifacts.")
PY
else
  existing_artifacts=("$REFERENCE_GGUF" "$Q8_GGUF" "$Q4_GGUF")
  [[ ! -e "$Q5_GGUF" && "$make_q5" == 0 ]] || existing_artifacts+=("$Q5_GGUF")
  "$PYTHON_BIN" - "${existing_artifacts[@]}" <<'PY'
import pathlib
import sys

for value in sys.argv[1:]:
    path = pathlib.Path(value)
    if not path.exists():
        continue
    if not path.is_file() or path.stat().st_size <= 4:
        raise SystemExit(f"refusing incomplete existing artifact: {path.name}")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise SystemExit(f"artifact does not have GGUF magic: {path.name}")
print("[convert_lfm25_gguf.sh] Existing unmanifested GGUF magic checks passed.")
PY
fi

create_reference() {
  if [[ -s "$REFERENCE_GGUF" ]]; then
    printf '[%s] Keeping existing reference: %s\n' "$SCRIPT_NAME" "$REFERENCE_GGUF"
    return
  fi
  [[ ! -e "$REFERENCE_GGUF" ]] || die "Refusing to overwrite incomplete artifact: $REFERENCE_GGUF"
  local partial="${REFERENCE_GGUF}.partial-$$.gguf"
  printf '[%s] Creating %s reference.\n' "$SCRIPT_NAME" "$REF_LABEL"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" "$CONVERTER" \
    --outtype "$reference_type" \
    --outfile "$partial" \
    "$INPUT_DIR"
  [[ -s "$partial" ]] || die "Converter did not create a nonempty artifact: $partial"
  mv --no-clobber "$partial" "$REFERENCE_GGUF"
  [[ ! -e "$partial" ]] \
    || die "Destination appeared concurrently; partial artifact was preserved: $partial"
}

create_quantized() {
  local quant_type="$1"
  local output="$2"
  if [[ -s "$output" ]]; then
    printf '[%s] Keeping existing %s artifact: %s\n' "$SCRIPT_NAME" "$quant_type" "$output"
    return
  fi
  [[ ! -e "$output" ]] || die "Refusing to overwrite incomplete artifact: $output"
  local partial="${output}.partial-$$.gguf"
  printf '[%s] Creating %s.\n' "$SCRIPT_NAME" "$quant_type"
  "$QUANTIZE" "$REFERENCE_GGUF" "$partial" "$quant_type"
  [[ -s "$partial" ]] || die "Quantizer did not create a nonempty artifact: $partial"
  mv --no-clobber "$partial" "$output"
  [[ ! -e "$partial" ]] \
    || die "Destination appeared concurrently; partial artifact was preserved: $partial"
}

create_reference
create_quantized Q8_0 "$Q8_GGUF"
create_quantized Q4_K_M "$Q4_GGUF"
if ((make_q5)); then
  create_quantized Q5_K_M "$Q5_GGUF"
fi

artifacts_to_check=("$REFERENCE_GGUF" "$Q8_GGUF" "$Q4_GGUF")
[[ ! -s "$Q5_GGUF" ]] || artifacts_to_check+=("$Q5_GGUF")
"$PYTHON_BIN" - "${artifacts_to_check[@]}" <<'PY'
import pathlib
import sys

for value in sys.argv[1:]:
    path = pathlib.Path(value)
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"artifact does not have GGUF magic: {path.name}")
PY

# Generate exactly one token from a fixed synthetic prompt. All CLI output is
# suppressed so neither prompts nor generated text can enter logs.
timeout_seconds="${LLAMA_CPP_SMOKE_TIMEOUT_SECONDS:-180}"
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] \
  || die "Invalid LLAMA_CPP_SMOKE_TIMEOUT_SECONDS: $timeout_seconds"
if ! timeout "$timeout_seconds" "$CLI" \
    --model "$Q4_GGUF" \
    --gpu-layers all \
    --ctx-size 256 \
    --batch-size 64 \
    --predict 1 \
    --seed 1 \
    --temperature 0 \
    --no-conversation \
    --single-turn \
    --simple-io \
    --no-display-prompt \
    --log-disable \
    --prompt 'Synthetic local smoke check.' \
    >/dev/null 2>&1; then
  die 'Pinned llama-cli could not load and generate from the Q4_K_M artifact.'
fi

"$PYTHON_BIN" - \
  "$MANIFEST" "$LOCK_FILE" "$INPUT_DIR" "$OUTPUT_PREFIX" "$reference_type" \
  "$PINNED_COMMIT" "$PINNED_TREE" "$allow_portable_binaries" \
  "$CLI" "$QUANTIZE" "$BENCH" \
  "$REFERENCE_GGUF" "$Q8_GGUF" "$Q4_GGUF" "$Q5_GGUF" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    manifest_path,
    lock_path,
    input_dir,
    output_prefix,
    reference_type,
    commit,
    tree,
    allow_portable,
    cli_path,
    quantize_path,
    bench_path,
    reference_path,
    q8_path,
    q4_path,
    q5_path,
) = sys.argv[1:]

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

lock = json.load(open(lock_path, encoding="utf-8"))
tool_paths = {
    "llama-cli": pathlib.Path(cli_path),
    "llama-quantize": pathlib.Path(quantize_path),
    "llama-bench": pathlib.Path(bench_path),
}
tool_hashes = {name: sha256_file(path) for name, path in tool_paths.items()}
locked_tool_hashes = lock["build"]["binary_sha256_observed"]
binary_hash_policy = "lock_exact"
if tool_hashes != locked_tool_hashes:
    if allow_portable != "1":
        raise SystemExit("llama.cpp binary hashes changed during conversion")
    binary_hash_policy = "portable_explicit_override"

input_root = pathlib.Path(input_dir)
input_hashes = {
    path.name: sha256_file(path)
    for path in sorted(input_root.iterdir())
    if path.is_file() and path.suffix in {".json", ".jinja", ".model", ".safetensors"}
}
artifacts = {}
for label, value in ((reference_type.upper(), reference_path), ("Q8_0", q8_path), ("Q4_K_M", q4_path)):
    path = pathlib.Path(value)
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise SystemExit(f"artifact does not have GGUF magic: {path.name}")
    artifacts[label] = {
        "file": path.name,
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
    }
if pathlib.Path(q5_path).is_file() and pathlib.Path(q5_path).stat().st_size > 0:
    path = pathlib.Path(q5_path)
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise SystemExit(f"artifact does not have GGUF magic: {path.name}")
    artifacts["Q5_K_M"] = {
        "file": path.name, "bytes": os.path.getsize(path), "sha256": sha256_file(path)
    }

manifest = {
    "schema_version": 2,
    "local_only": True,
    "input_directory_name": pathlib.Path(input_dir).name,
    "input_config_sha256": input_hashes["config.json"],
    "input_files_sha256": input_hashes,
    "output_prefix_name": pathlib.Path(output_prefix).name,
    "llama_cpp_commit": commit,
    "llama_cpp_tree": tree,
    "llama_cpp_binary_sha256": tool_hashes,
    "binary_hash_policy": binary_hash_policy,
    "artifacts": artifacts,
    "verification": {
        "gguf_magic": {
            "status": "passed",
            "artifacts": sorted(artifacts),
        },
        "load_and_generation_smoke": {
            "status": "passed",
            "artifact": pathlib.Path(q4_path).name,
            "tool": "llama-cli",
            "gpu_layers": "all",
            "generated_tokens": 1,
            "input": "fixed synthetic text",
            "prompt_and_output_logging": "suppressed",
        },
        "other_artifact_load_smokes": "not_run_by_this_conversion_script",
    },
}
manifest_file = pathlib.Path(manifest_path)
temporary = manifest_file.with_name(f".{manifest_file.name}.{os.getpid()}.tmp")
try:
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_file)
finally:
    if temporary.exists():
        temporary.unlink()
PY

printf '[%s] Conversion and Q4_K_M load/generation smoke check passed.\n' "$SCRIPT_NAME"
printf '[%s] Local manifest: %s\n' "$SCRIPT_NAME" "$MANIFEST"
