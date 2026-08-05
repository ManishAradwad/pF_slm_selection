#!/usr/bin/env python3
"""Run pinned llama-bench for true prompt-processing and decode throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = (
    REPO_ROOT / "UPSTREAM" / "llama.cpp" / "build-lfm25-cuda" / "bin" / "llama-bench"
)
DEFAULT_LOCK = REPO_ROOT / "configs" / "lfm25" / "llama_cpp.lock.json"
AGGREGATE_RESULT_FIELDS = (
    "avg_ns",
    "avg_ts",
    "backends",
    "build_commit",
    "build_number",
    "cpu_info",
    "cpu_mask",
    "cpu_strict",
    "devices",
    "embeddings",
    "fit_min_ctx",
    "fit_target",
    "flash_attn",
    "gpu_info",
    "load_mode",
    "main_gpu",
    "model_n_params",
    "model_size",
    "model_type",
    "n_batch",
    "n_cpu_moe",
    "n_depth",
    "n_gen",
    "n_gpu_layers",
    "n_prompt",
    "n_threads",
    "n_ubatch",
    "no_host",
    "no_kv_offload",
    "no_op_offload",
    "poll",
    "split_mode",
    "stddev_ns",
    "stddev_ts",
    "tensor_buft_overrides",
    "tensor_split",
    "test_time",
    "type_k",
    "type_v",
)
RUNTIME_IDENTIFIER_FIELDS = (
    "backends",
    "build_commit",
    "build_number",
    "cpu_info",
    "gpu_info",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_spec(value: str) -> tuple[str, Path]:
    label, separator, path_text = value.partition("=")
    if not separator or not label or not path_text:
        raise argparse.ArgumentTypeError("GGUF must use LABEL=PATH")
    if not all(character.isalnum() or character in "_-" for character in label):
        raise argparse.ArgumentTypeError("GGUF label must be alphanumeric with _ or -")
    return label, Path(path_text)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _aggregate_result(row: dict[str, Any]) -> dict[str, Any]:
    """Retain aggregate metrics and declared run metadata, never raw samples."""
    return {field: row[field] for field in AGGREGATE_RESULT_FIELDS if field in row}


def _runtime_identifiers(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in RUNTIME_IDENTIFIER_FIELDS
        if field in row
    }


def _validate_results(
    rows: Any,
    *,
    commit: str,
    prompt_tokens: int,
    generation_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError("llama-bench did not return exactly prompt and generation rows")
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("llama-bench returned malformed JSON")
    prompt_rows = [row for row in rows if row.get("n_prompt") == prompt_tokens and row.get("n_gen") == 0]
    generation_rows = [
        row for row in rows if row.get("n_prompt") == 0 and row.get("n_gen") == generation_tokens
    ]
    if len(prompt_rows) != 1 or len(generation_rows) != 1:
        raise RuntimeError("llama-bench JSON does not match the requested token counts")
    reported_commits = [str(row.get("build_commit", "")) for row in rows]
    if any(
        len(reported) < 7 or not commit.startswith(reported)
        for reported in reported_commits
    ):
        raise RuntimeError("llama-bench result does not match the pinned commit")
    return prompt_rows[0], generation_rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", action="append", required=True, type=_model_spec)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "RESULTS" / "lfm25" / "benchmarks" / "llama_bench.json",
    )
    parser.add_argument("--prompt-tokens", type=int, default=214)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--gpu-layers", type=int, default=99)
    args = parser.parse_args()

    if min(args.prompt_tokens, args.generation_tokens, args.repetitions) <= 0:
        raise SystemExit("token counts and repetitions must be positive")
    lock_path = args.lock.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    commit = str(lock["upstream"]["commit"])
    tree = str(lock["upstream"]["tree"])
    repository = str(lock["upstream"]["repository"])
    expected_hash = str(lock["build"]["binary_sha256_observed"]["llama-bench"])
    toolchain_environment = lock["build"].get("toolchain_observed")
    if not isinstance(toolchain_environment, dict):
        raise RuntimeError("llama.cpp lock has no toolchain environment mapping")
    bench = args.bench.resolve(strict=True)
    bench_hash = _sha256(bench)
    if bench_hash != expected_hash:
        raise RuntimeError("llama-bench binary hash does not match the lock")
    script_path = Path(__file__).resolve(strict=True)
    script_hash = _sha256(script_path)
    lock_hash = _sha256(lock_path)

    variants: dict[str, Any] = {}
    concise: dict[str, Any] = {}
    runtime_environments: dict[str, Any] = {}
    for label, model_path in args.gguf:
        if label in variants:
            raise ValueError(f"duplicate GGUF label: {label}")
        model = model_path.resolve(strict=True)
        command = [
            str(bench),
            "--offline",
            "--model",
            str(model),
            "--n-prompt",
            str(args.prompt_tokens),
            "--n-gen",
            str(args.generation_tokens),
            "--repetitions",
            str(args.repetitions),
            "--n-gpu-layers",
            str(args.gpu_layers),
            "--output",
            "json",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(completed.stdout)
        prompt, generation = _validate_results(
            rows,
            commit=commit,
            prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens,
        )
        model_hash = _sha256(model)
        variants[label] = {
            "file": str(model),
            "file_size_bytes": model.stat().st_size,
            "file_sha256": model_hash,
            "invocation": {
                "argv": command,
                "cwd": str(REPO_ROOT.resolve(strict=True)),
            },
            "prompt_processing": _aggregate_result(prompt),
            "token_generation": _aggregate_result(generation),
        }
        runtime_environments[label] = _runtime_identifiers(prompt)
        concise[label] = {
            "prompt_tokens_per_second": round(float(prompt["avg_ts"]), 3),
            "prompt_tokens_per_second_stddev": round(float(prompt["stddev_ts"]), 3),
            "decode_tokens_per_second": round(float(generation["avg_ts"]), 3),
            "decode_tokens_per_second_stddev": round(
                float(generation["stddev_ts"]), 3
            ),
        }

    report = {
        "schema_version": 2,
        "llama_cpp_commit": commit,
        "llama_cpp_tree": tree,
        "llama_cpp_lock_sha256": lock_hash,
        "llama_bench_sha256": expected_hash,
        "configuration": {
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "repetitions": args.repetitions,
            "gpu_layers": args.gpu_layers,
            "offline": True,
            "output_format": "json",
            "workload": {
                "prompt_tokens": args.prompt_tokens,
                "generation_tokens": args.generation_tokens,
            },
            "offload": {"n_gpu_layers": args.gpu_layers},
        },
        "provenance": {
            "benchmark_script": {
                "path": str(script_path),
                "bytes": script_path.stat().st_size,
                "sha256": script_hash,
            },
            "llama_bench": {
                "path": str(bench),
                "bytes": bench.stat().st_size,
                "sha256": bench_hash,
                "pinned_sha256": expected_hash,
            },
            "llama_cpp": {
                "repository": repository,
                "commit": commit,
                "tree": tree,
                "lock": {
                    "path": str(lock_path),
                    "bytes": lock_path.stat().st_size,
                    "sha256": lock_hash,
                },
            },
            "environment": {
                "toolchain_observed_from_lock": toolchain_environment,
                "llama_bench_reported_by_variant": runtime_environments,
            },
        },
        "variants": variants,
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "throughput": concise}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
