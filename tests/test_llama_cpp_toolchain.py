import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "lfm25" / "llama_cpp.lock.json"
SETUP_PATH = ROOT / "scripts" / "setup_lfm25_llama_cpp.sh"
CONVERT_PATH = ROOT / "scripts" / "convert_lfm25_gguf.sh"
EXPECTED_TREE = "3d5eccf08dbfe6edab1939314964475a603ab2de"
REQUIRED_BINARIES = {"llama-cli", "llama-quantize", "llama-bench"}


def _lock() -> dict[str, object]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_lock_records_exact_tree_and_all_observed_binary_hashes() -> None:
    lock = _lock()

    assert lock["upstream"]["tree"] == EXPECTED_TREE
    hashes = lock["build"]["binary_sha256_observed"]
    assert set(hashes) == REQUIRED_BINARIES
    for digest in hashes.values():
        assert len(digest) == 64
        int(digest, 16)


def test_lock_describes_only_the_load_smoke_that_was_run() -> None:
    verification = _lock()["verification"]

    assert "smoke_conversion_quantization_and_load" not in verification
    smoke = verification["load_and_generation_smoke"]
    assert smoke["status"] == "passed"
    assert smoke["artifact"] == "Q4_K_M"
    assert smoke["generated_tokens"] == 1
    assert verification["other_artifact_load_smokes"].startswith("not_run")


def test_setup_validates_tree_sources_and_binaries_with_explicit_portable_opt_in() -> None:
    source = SETUP_PATH.read_text(encoding="utf-8")
    help_output = subprocess.run(
        ["bash", str(SETUP_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "lock_get upstream.tree" in source
    assert "rev-parse 'HEAD^{tree}'" in source
    assert 'lock["source_sha256"]' in source
    assert 'lock["build"]["binary_sha256_observed"]' in source
    assert "required = (\"llama-cli\", \"llama-quantize\", \"llama-bench\")" in source
    assert "built binary hash mismatch" in source
    assert "--allow-portable-binaries" in help_output
    assert "exact hashes" in help_output.lower()


def test_conversion_validates_tools_before_reuse_and_hashes_every_gguf() -> None:
    source = CONVERT_PATH.read_text(encoding="utf-8")
    help_output = subprocess.run(
        ["bash", str(CONVERT_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    binary_check = source.index('expected_hashes = lock["build"]["binary_sha256_observed"]')
    manifest_reuse = source.index('if [[ -f "$MANIFEST" ]]')
    artifact_reuse = source.index("create_reference()")
    assert binary_check < manifest_reuse < artifact_reuse
    assert "rev-parse 'HEAD^{tree}'" in source
    assert "--allow-portable-binaries" in help_output

    assert 'manifest.get("schema_version") != 2' in source
    assert 'record.get("bytes")' in source
    assert 'record.get("sha256")' in source
    assert "path.stat().st_size != expected_size" in source
    assert "sha256_file(path) != expected_hash" in source
    assert 'handle.read(4) != b"GGUF"' in source
    assert '"sha256": sha256_file(path)' in source
    assert '"schema_version": 2' in source


def test_conversion_manifest_smoke_evidence_is_narrow_and_atomic() -> None:
    source = CONVERT_PATH.read_text(encoding="utf-8")

    smoke_command = source.index('if ! timeout "$timeout_seconds" "$CLI"')
    manifest_write = source.index("os.replace(temporary, manifest_file)")
    assert smoke_command < manifest_write
    assert '"load_and_generation_smoke": {' in source
    assert '"artifact": pathlib.Path(q4_path).name' in source
    assert '"generated_tokens": 1' in source
    assert '"other_artifact_load_smokes": "not_run_by_this_conversion_script"' in source
