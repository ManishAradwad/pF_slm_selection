import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts import benchmark_lfm25_gguf as benchmark


COMMIT = "a" * 40
TREE = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(*, prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "avg_ns": 123,
        "avg_ts": 456.75,
        "backends": "CUDA",
        "build_commit": COMMIT[:9],
        "build_number": 10252,
        "cpu_info": "Test CPU",
        "gpu_info": "Test GPU",
        "model_filename": "/raw/path/that/is-not-copied.gguf",
        "n_gen": generation_tokens,
        "n_gpu_layers": 99,
        "n_prompt": prompt_tokens,
        "samples_ns": [100, 146],
        "samples_ts": [400.0, 513.5],
        "stddev_ns": 23,
        "stddev_ts": 56.75,
        "unexpected_raw_field": "must not be persisted",
    }


def test_benchmark_report_has_direct_provenance_and_only_aggregate_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bench_path = tmp_path / "llama-bench"
    bench_path.write_bytes(b"pinned llama-bench test binary")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF test artifact")
    output_path = tmp_path / "benchmark.json"
    lock_path = tmp_path / "llama_cpp.lock.json"
    lock = {
        "upstream": {
            "repository": "https://example.invalid/llama.cpp.git",
            "commit": COMMIT,
            "tree": TREE,
        },
        "build": {
            "binary_sha256_observed": {"llama-bench": _sha256(bench_path)},
            "toolchain_observed": {
                "environment": "native WSL2; test distribution",
                "kernel": "test-kernel",
                "gpu": "Test GPU",
            },
        },
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    rows = [
        _row(prompt_tokens=214, generation_tokens=0),
        _row(prompt_tokens=0, generation_tokens=32),
    ]
    observed_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed_commands.append(command)
        assert kwargs["cwd"] == benchmark.REPO_ROOT
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(rows), stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(benchmark.__file__)),
            "--gguf",
            f"Q4={model_path}",
            "--bench",
            str(bench_path),
            "--lock",
            str(lock_path),
            "--output",
            str(output_path),
            "--prompt-tokens",
            "214",
            "--generation-tokens",
            "32",
            "--repetitions",
            "5",
            "--gpu-layers",
            "99",
        ],
    )

    assert benchmark.main() == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == 2
    assert report["llama_cpp_commit"] == COMMIT
    assert report["llama_cpp_tree"] == TREE
    assert report["llama_cpp_lock_sha256"] == _sha256(lock_path)
    assert report["llama_bench_sha256"] == _sha256(bench_path)

    provenance = report["provenance"]
    assert provenance["benchmark_script"] == {
        "path": str(Path(benchmark.__file__).resolve()),
        "bytes": Path(benchmark.__file__).stat().st_size,
        "sha256": _sha256(Path(benchmark.__file__)),
    }
    assert provenance["llama_bench"] == {
        "path": str(bench_path.resolve()),
        "bytes": bench_path.stat().st_size,
        "sha256": _sha256(bench_path),
        "pinned_sha256": _sha256(bench_path),
    }
    assert provenance["llama_cpp"]["commit"] == COMMIT
    assert provenance["llama_cpp"]["tree"] == TREE
    assert provenance["llama_cpp"]["lock"]["sha256"] == _sha256(lock_path)
    assert provenance["environment"]["toolchain_observed_from_lock"] == (
        lock["build"]["toolchain_observed"]
    )
    assert provenance["environment"]["llama_bench_reported_by_variant"]["Q4"] == {
        "backends": "CUDA",
        "build_commit": COMMIT[:9],
        "build_number": 10252,
        "cpu_info": "Test CPU",
        "gpu_info": "Test GPU",
    }

    assert report["configuration"] == {
        "generation_tokens": 32,
        "gpu_layers": 99,
        "offline": True,
        "offload": {"n_gpu_layers": 99},
        "output_format": "json",
        "prompt_tokens": 214,
        "repetitions": 5,
        "workload": {"generation_tokens": 32, "prompt_tokens": 214},
    }
    variant = report["variants"]["Q4"]
    assert variant["file"] == str(model_path.resolve())
    assert variant["file_size_bytes"] == model_path.stat().st_size
    assert variant["file_sha256"] == _sha256(model_path)
    expected_command = [
        str(bench_path.resolve()),
        "--offline",
        "--model",
        str(model_path.resolve()),
        "--n-prompt",
        "214",
        "--n-gen",
        "32",
        "--repetitions",
        "5",
        "--n-gpu-layers",
        "99",
        "--output",
        "json",
    ]
    assert observed_commands == [expected_command]
    assert variant["invocation"] == {
        "argv": expected_command,
        "cwd": str(benchmark.REPO_ROOT.resolve()),
    }
    assert "samples_ns" not in variant["prompt_processing"]
    assert "samples_ts" not in variant["token_generation"]
    assert "unexpected_raw_field" not in variant["prompt_processing"]
    assert "model_filename" not in variant["prompt_processing"]
    assert variant["prompt_processing"]["avg_ts"] == 456.75
    assert variant["token_generation"]["stddev_ts"] == 56.75


def test_aggregate_result_uses_an_explicit_allowlist() -> None:
    aggregate = benchmark._aggregate_result(
        {
            "avg_ts": 12.5,
            "stddev_ts": 1.25,
            "samples_ts": [11.25, 13.75],
            "prompt": "sensitive input must never be copied",
            "output": "generated text must never be copied",
        }
    )

    assert aggregate == {"avg_ts": 12.5, "stddev_ts": 1.25}
