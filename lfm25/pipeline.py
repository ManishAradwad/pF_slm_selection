"""Configuration and command planning for the PocketFinancer model pipeline.

The public CLI deliberately orchestrates the existing, tested building blocks
instead of duplicating training or evaluation logic.  This module contains no
GPU imports, so plans can be validated in CI and on machines without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PIPELINE_SCHEMA_VERSION = 1
EXECUTION_STAGES = (
    "build-data",
    "train",
    "evaluate-hf",
    "merge",
    "convert",
    "evaluate-gguf",
)


class PipelineConfigError(ValueError):
    """Raised when a pipeline declaration is incomplete or inconsistent."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineConfigError(f"{name} must be numeric")
    return value


def load_pipeline_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineConfigError(f"could not read pipeline config {path}: {error}") from error
    config = dict(_mapping(value, "pipeline config"))
    validate_pipeline_config(config)
    return config


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise PipelineConfigError(
            f"schema_version must be {PIPELINE_SCHEMA_VERSION}"
        )
    _text(config.get("name"), "name")
    _text(config.get("app_profile"), "app_profile")

    model = _mapping(config.get("model"), "model")
    _text(model.get("id"), "model.id")
    _text(model.get("local_path"), "model.local_path")
    _text(model.get("lock"), "model.lock")
    if not isinstance(model.get("thinking_mode"), bool):
        raise PipelineConfigError("model.thinking_mode must be true or false")

    data = _mapping(config.get("data"), "data")
    for key in ("manifest", "output_dir", "train", "dev"):
        _text(data.get(key), f"data.{key}")
    builder = _mapping(data.get("builder"), "data.builder")
    for key in (
        "dev_fraction",
        "seed",
        "minimum_silver_confidence",
        "max_per_template",
        "max_per_category",
        "max_null_to_transaction_ratio",
    ):
        _number(builder.get(key), f"data.builder.{key}")

    training = _mapping(config.get("training"), "training")
    if training.get("method") != "lora":
        raise PipelineConfigError("training.method currently supports only 'lora'")
    for key in ("output_dir", "loss_mode"):
        _text(training.get(key), f"training.{key}")
    for key in (
        "rank",
        "alpha",
        "dropout",
        "learning_rate",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "max_length",
        "first_supervised_token_weight",
        "warmup_ratio",
        "weight_decay",
        "early_stopping_patience",
        "seed",
    ):
        _number(training.get(key), f"training.{key}")
    if training.get("prompt_profile") != "pocketfinancer":
        raise PipelineConfigError(
            "training.prompt_profile must be 'pocketfinancer' on the supported path"
        )

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    for key in ("dataset", "hf_output_dir", "gguf_output_dir"):
        _text(evaluation.get(key), f"evaluation.{key}")
    _number(evaluation.get("n_ctx"), "evaluation.n_ctx")
    if str(data["train"]) == str(evaluation["dataset"]):
        raise PipelineConfigError("the regression dataset must never be used as training data")
    export = _mapping(config.get("export"), "export")
    for key in ("merged_dir", "gguf_prefix", "deployable_gguf"):
        _text(export.get(key), f"export.{key}")


def _option(command: list[str], flag: str, value: Any) -> None:
    command.extend((flag, str(value)))


def _python(config: Mapping[str, Any]) -> str:
    runtime = config.get("runtime", {})
    if isinstance(runtime, Mapping):
        value = runtime.get("python")
        if isinstance(value, str) and value:
            return value
    return ".venv/bin/python"


def build_stage_commands(
    config: Mapping[str, Any], *, force_data: bool = False
) -> dict[str, list[str]]:
    """Return argv arrays for every reproducible pipeline stage."""

    validate_pipeline_config(config)
    python = _python(config)
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    builder = _mapping(data["builder"], "data.builder")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    export = _mapping(config["export"], "export")

    build_data = [python, "scripts/build_lfm25_private_sft_v2.py"]
    for flag, key in (
        ("--manifest", "manifest"),
        ("--output-dir", "output_dir"),
    ):
        _option(build_data, flag, data[key])
    for flag, key in (
        ("--dev-fraction", "dev_fraction"),
        ("--seed", "seed"),
        ("--minimum-silver-confidence", "minimum_silver_confidence"),
        ("--max-per-template", "max_per_template"),
        ("--max-per-category", "max_per_category"),
        ("--max-null-to-transaction-ratio", "max_null_to_transaction_ratio"),
    ):
        _option(build_data, flag, builder[key])
    if force_data:
        build_data.append("--force")

    train = [python, "scripts/train_lfm25_lora.py"]
    for flag, value in (
        ("--model", model["local_path"]),
        ("--train", data["train"]),
        ("--eval", data["dev"]),
        ("--output-dir", training["output_dir"]),
        ("--prompt-profile", "pocketfinancer"),
        ("--rank", training["rank"]),
        ("--alpha", training["alpha"]),
        ("--dropout", training["dropout"]),
        ("--learning-rate", training["learning_rate"]),
        ("--epochs", training["epochs"]),
        ("--batch-size", training["batch_size"]),
        ("--eval-batch-size", training["eval_batch_size"]),
        ("--gradient-accumulation", training["gradient_accumulation"]),
        ("--max-length", training["max_length"]),
        ("--loss-mode", training["loss_mode"]),
        (
            "--first-supervised-token-weight",
            training["first_supervised_token_weight"],
        ),
        ("--warmup-ratio", training["warmup_ratio"]),
        ("--weight-decay", training["weight_decay"]),
        ("--early-stopping-patience", training["early_stopping_patience"]),
        ("--seed", training["seed"]),
    ):
        _option(train, flag, value)

    adapter = f"{training['output_dir']}/adapter"
    evaluate_hf = [python, "scripts/evaluate_lfm25_android_hf.py"]
    for flag, value in (
        ("--model", model["local_path"]),
        ("--adapter", adapter),
        ("--dataset", evaluation["dataset"]),
        ("--output-dir", evaluation["hf_output_dir"]),
        ("--contract", "pocketfinancer"),
        ("--n-ctx", evaluation["n_ctx"]),
        ("--max-new-tokens", 256),
        ("--seed", training["seed"]),
    ):
        _option(evaluate_hf, flag, value)

    merge = [python, "scripts/merge_lfm25_lora.py"]
    for flag, value in (
        ("--base-model", model["local_path"]),
        ("--adapter", adapter),
        ("--output-dir", export["merged_dir"]),
    ):
        _option(merge, flag, value)

    convert = [
        "bash",
        "scripts/convert_lfm25_gguf.sh",
        str(export["merged_dir"]),
        str(export["gguf_prefix"]),
    ]

    evaluate_gguf = [python, "scripts/evaluate_lfm25_android_gguf.py"]
    for flag, value in (
        ("--gguf", export["deployable_gguf"]),
        ("--tokenizer", model["local_path"]),
        ("--dataset", evaluation["dataset"]),
        ("--output-dir", evaluation["gguf_output_dir"]),
        ("--contract", "pocketfinancer"),
        ("--thinking-mode", "on" if model["thinking_mode"] else "off"),
        ("--seed", training["seed"]),
    ):
        _option(evaluate_gguf, flag, value)
    if bool(evaluation.get("grammar", False)):
        _option(evaluate_gguf, "--grammar", evaluation.get("grammar_path", "DATA/sms_extraction.gbnf"))
        evaluate_gguf.append("--with-grammar")
    else:
        evaluate_gguf.append("--no-grammar")

    return {
        "build-data": build_data,
        "train": train,
        "evaluate-hf": evaluate_hf,
        "merge": merge,
        "convert": convert,
        "evaluate-gguf": evaluate_gguf,
    }


@dataclass(frozen=True)
class InputRequirement:
    label: str
    path: Path


def stage_requirements(
    config: Mapping[str, Any], stage: str, repo_root: Path
) -> Sequence[InputRequirement]:
    """Return files/directories that must exist before a stage starts."""

    commands = build_stage_commands(config)
    if stage not in commands:
        raise PipelineConfigError(f"unknown stage: {stage}")
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    export = _mapping(config["export"], "export")

    common = [
        InputRequirement("app_profile", repo_root / str(config["app_profile"])),
        InputRequirement("model_lock", repo_root / str(model["lock"])),
    ]
    if stage == "build-data":
        return [*common, InputRequirement("manifest", repo_root / str(data["manifest"]))]
    if stage == "train":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("train", repo_root / str(data["train"])),
            InputRequirement("dev", repo_root / str(data["dev"])),
        ]
    if stage == "evaluate-hf":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("adapter", repo_root / str(training["output_dir"]) / "adapter"),
            InputRequirement("dataset", repo_root / str(evaluation["dataset"])),
        ]
    if stage == "merge":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("adapter", repo_root / str(training["output_dir"]) / "adapter"),
        ]
    if stage == "convert":
        return [*common, InputRequirement("merged_model", repo_root / str(export["merged_dir"]))]
    return [
        *common,
        InputRequirement("gguf", repo_root / str(export["deployable_gguf"])),
        InputRequirement("dataset", repo_root / str(evaluation["dataset"])),
    ]


def missing_requirements(
    config: Mapping[str, Any], stage: str, repo_root: Path
) -> list[InputRequirement]:
    return [item for item in stage_requirements(config, stage, repo_root) if not item.path.exists()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_model(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Verify the local HF base against the immutable checked-in model lock."""

    validate_pipeline_config(config)
    model = _mapping(config["model"], "model")
    lock_path = repo_root / str(model["lock"])
    model_root = repo_root / str(model["local_path"])
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineConfigError(f"could not read model lock {lock_path}: {error}") from error
    locked_model = _mapping(_mapping(lock, "model lock").get("model"), "model lock.model")
    if locked_model.get("repo") != model.get("id"):
        raise PipelineConfigError("pipeline model.id does not match model lock repo")
    files = _mapping(locked_model.get("files"), "model lock.model.files")
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in sorted(files.items()):
        relative_name = _text(relative, "locked model filename")
        expected_hash = _text(expected, f"model lock hash for {relative_name}")
        path = model_root / relative_name
        if not path.is_file():
            missing.append(relative_name)
        elif _sha256(path) != expected_hash:
            mismatched.append(relative_name)
    if missing or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if mismatched:
            details.append("hash_mismatch=" + ",".join(mismatched))
        raise PipelineConfigError("local base model does not match lock: " + "; ".join(details))
    return {
        "model_id": model["id"],
        "revision": locked_model.get("revision"),
        "verified_files": len(files),
        "lock": str(lock_path.relative_to(repo_root)),
    }
