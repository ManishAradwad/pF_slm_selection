import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_lfm25_llama_cpp.sh"
CONVERT = ROOT / "scripts" / "convert_lfm25_gguf.sh"
LOCK = ROOT / "configs" / "lfm25" / "llama_cpp.lock.json"


def test_llama_cpp_lock_pins_official_full_revision_and_lfm2_support() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert lock["upstream"]["repository"] == "https://github.com/ggml-org/llama.cpp.git"
    assert len(lock["upstream"]["commit"]) == 40
    int(lock["upstream"]["commit"], 16)
    assert "Lfm2ForCausalLM" in lock["lfm2_support"]["hf_architectures"]
    assert lock["lfm2_support"]["gguf_architecture"] == "lfm2"
    assert lock["build"]["options"]["GGML_CUDA"] == "ON"
    assert set(lock["build"]["targets"]) == {"llama-cli", "llama-quantize", "llama-bench"}
    assert "conversion/lfm2.py" in lock["source_sha256"]
    assert "tools/quantize/quantize.cpp" in lock["source_sha256"]


def test_llama_cpp_shell_scripts_are_valid_bash() -> None:
    for script in (SETUP, CONVERT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_setup_is_pinned_cuda_native_wsl_and_push_disabled() -> None:
    source = SETUP.read_text(encoding="utf-8")

    assert "configs/lfm25/llama_cpp.lock.json" in source
    assert "https://github.com/ggml-org/llama.cpp.git" in source
    assert "git clone --filter=blob:none --no-tags" in source
    assert "remote set-url --push origin DISABLED" in source
    assert "GGML_CUDA=ON" in source
    assert "llama-cli llama-quantize llama-bench" in source
    assert "/proc/sys/kernel/osrelease" in source
    assert "/.dockerenv" in source
    assert "flock" in source


def test_conversion_contract_is_local_idempotent_and_complete() -> None:
    source = CONVERT.read_text(encoding="utf-8")

    assert "--remote" not in source
    assert "Q8_0" in source
    assert "Q4_K_M" in source
    assert "Q5_K_M" in source
    assert "--outtype \"$reference_type\"" in source
    assert "Keeping existing" in source
    assert "Refusing to overwrite incomplete artifact" in source
    assert "mv --no-clobber" in source
    assert "input_files_sha256" in source
    assert source.index("flock 8") < source.index('if [[ -f "$MANIFEST" ]]')
    assert "HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "--predict 1" in source
    assert "--log-disable" in source
    assert ">/dev/null 2>&1" in source
    for forbidden in ("curl ", "wget ", "git push", "hf upload", "huggingface-cli upload"):
        assert forbidden not in source.lower()
