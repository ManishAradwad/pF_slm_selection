#!/usr/bin/env python3
"""Deterministic PocketFinancer-aligned BF16 LoRA SFT.

The supported happy path serializes every example exactly like the Android app
and computes loss only on the assistant completion. Historical profiles remain
available solely to reproduce earlier experiments.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import stat
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.contract import parse_gold, parse_prediction  # noqa: E402
from lfm25.prompts import extraction_messages  # noqa: E402
from lfm25.prompts import PRODUCTION_SYSTEM_PROMPT  # noqa: E402
from lfm25.provenance import fingerprint_named_files  # noqa: E402
from lfm25.training_loss import LOSS_MODE, normalized_completion_cross_entropy  # noqa: E402
from lfm25.training_provenance import (  # noqa: E402
    RUN_MANIFEST_FORMAT,
    adapter_artifact_evidence,
    trainer_code_fingerprints,
    training_manifest_artifact_binding,
)


LORA_TARGETS = ["in_proj", "q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"]
LEGACY_MAX_LENGTH = 512
ANDROID_CONTEXT_LENGTH = 3072
ANDROID_OUTER_SYSTEM_PROMPT = "You are a helpful financial SMS extraction assistant."
PROMPT_PROFILE_ALIASES = {
    "legacy": "legacy",
    "legacy_short": "legacy",
    "short": "legacy",
    "android": "android",
    "pocketfinancer": "android",
    "pocketfinancer_android": "android",
    "candidate_selector": "candidate_selector",
    "selector": "candidate_selector",
}
HF_CONFIG_ASSETS = (
    "config.json",
    "generation_config.json",
)
HF_TOKENIZER_ASSETS = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)
HF_WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
)
HF_WEIGHT_INDEX_PATTERNS = (
    "*.safetensors.index.json",
    "pytorch_model*.bin.index.json",
)
RESUME_PROVENANCE_FILENAME = "resume_provenance.json"
RESUME_PROVENANCE_FORMAT = "lfm25_lora_resume_provenance_v2"
CHECKPOINT_ARTIFACT_FORMAT = "lfm25_checkpoint_artifact_v1"
CHECKPOINT_RESUME_BINDING_FORMAT = "lfm25_checkpoint_resume_binding_v1"
MODEL_LOCK_PATH = REPO_ROOT / "configs" / "lfm25" / "model.lock.json"
_CHECKPOINT_DIRECTORY_RE = re.compile(r"checkpoint-[1-9]\d*")
_CHECKPOINT_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CHECKPOINT_WEIGHT_RE = re.compile(
    r"(?:adapter_model|model|pytorch_model)"
    r"(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)"
)
_CHECKPOINT_RNG_RE = re.compile(r"rng_state(?:_\d+)?\.pth")
_REQUIRED_CHECKPOINT_FILES = frozenset(
    {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path, *, filename: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "filename": filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _candidate_dataset_report_provenance(
    train_path: Path,
    eval_path: Path,
    report_path: Path | None,
) -> dict[str, Any] | None:
    expected_train = "candidate_protocol_v1_train.jsonl"
    expected_eval = "candidate_protocol_v1_dev.jsonl"
    observes_candidate_name = train_path.name == expected_train or eval_path.name == expected_eval
    if not observes_candidate_name:
        if report_path is not None:
            raise RuntimeError("--dataset-report requires the Candidate V1 train/dev bundle")
        return None
    if (
        train_path.name != expected_train
        or eval_path.name != expected_eval
        or train_path.resolve(strict=True).parent != eval_path.resolve(strict=True).parent
    ):
        raise RuntimeError("Candidate V1 train/dev inputs are not a matched dataset bundle")
    if report_path is None:
        raise RuntimeError("--dataset-report is required for Candidate V1 training")
    resolved_report = report_path.resolve(strict=True)
    if (
        resolved_report.name != "candidate_protocol_v1_report.json"
        or resolved_report.parent != train_path.resolve(strict=True).parent
    ):
        raise RuntimeError("Candidate V1 dataset report must belong to the train/dev bundle")
    return _file_fingerprint(
        resolved_report,
        filename="candidate_protocol_v1_report.json",
    )


def _locked_model_provenance(
    base_model_provenance: dict[str, Any],
    model_lock_path: Path,
) -> dict[str, Any]:
    resolved_lock = model_lock_path.resolve(strict=True)
    if resolved_lock.name != "model.lock.json":
        raise RuntimeError("--model-lock must name model.lock.json")
    lock_bytes = resolved_lock.read_bytes()
    try:
        lock = json.loads(lock_bytes)
        model = lock["model"]
        locked_files = model["files"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("LFM2.5 model lock is malformed") from error
    if model.get("repo") != "LiquidAI/LFM2.5-350M" or not isinstance(locked_files, dict):
        raise RuntimeError("LFM2.5 model lock has an unexpected identity")
    runtime_locked_files = {
        name: digest
        for name, digest in locked_files.items()
        if name not in {"LICENSE", "README.md"}
    }
    observed_files = base_model_provenance.get("files")
    if not isinstance(observed_files, dict) or set(observed_files) != set(runtime_locked_files):
        raise RuntimeError("local base-model files differ from the immutable model lock")
    for name, evidence in observed_files.items():
        if not isinstance(evidence, dict) or runtime_locked_files.get(name) != evidence.get(
            "sha256"
        ):
            raise RuntimeError("local base model differs from the immutable model lock")
    return {
        "filename": "model.lock.json",
        "bytes": len(lock_bytes),
        "sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }


def _require_unchanged(label: str, expected: Any, observed: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} changed while training; refusing provenance output")


def target_module_inventory(model: Any) -> dict[str, Any]:
    """Return aggregate LoRA leaf-name coverage and reject an empty match."""

    counts = Counter(
        name.rsplit(".", 1)[-1]
        for name, _module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in LORA_TARGETS
    )
    inventory = {
        "configured_leaf_names": list(LORA_TARGETS),
        "matched_module_count": sum(counts.values()),
        "matched_leaf_counts": {name: counts.get(name, 0) for name in LORA_TARGETS},
    }
    if inventory["matched_module_count"] <= 0:
        raise RuntimeError("LoRA target inspection found no matching modules")
    return inventory


def _resolve_local_hf_root(model_spec: str | Path) -> Path:
    candidate = Path(model_spec).expanduser()
    if candidate.is_dir():
        return candidate.resolve(strict=True)

    try:
        from transformers.utils import cached_file

        cached_config = cached_file(
            str(model_spec),
            "config.json",
            local_files_only=True,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot resolve local Hugging Face model snapshot: {model_spec}"
        ) from error
    if cached_config is None:
        raise RuntimeError(f"cannot resolve local Hugging Face model snapshot: {model_spec}")
    return Path(cached_config).parent.resolve(strict=True)


def _local_hf_model_provenance(model_spec: str | Path) -> dict[str, Any]:
    """Fingerprint the exact local weights, indexes, config, and tokenizer assets."""

    model_root = _resolve_local_hf_root(model_spec)
    config_files = [name for name in HF_CONFIG_ASSETS if (model_root / name).is_file()]
    tokenizer_files = [name for name in HF_TOKENIZER_ASSETS if (model_root / name).is_file()]
    weight_files = sorted(
        {
            path.name
            for pattern in HF_WEIGHT_PATTERNS
            for path in model_root.glob(pattern)
            if path.is_file()
        }
    )
    weight_index_files = sorted(
        {
            path.name
            for pattern in HF_WEIGHT_INDEX_PATTERNS
            for path in model_root.glob(pattern)
            if path.is_file()
        }
    )
    if "config.json" not in config_files:
        raise RuntimeError(f"local Hugging Face model has no config.json: {model_root}")
    if not tokenizer_files:
        raise RuntimeError(f"local Hugging Face model has no tokenizer assets: {model_root}")
    if not weight_files:
        raise RuntimeError(f"local Hugging Face model has no weight files: {model_root}")

    names = sorted({*config_files, *tokenizer_files, *weight_files, *weight_index_files})
    evidence = fingerprint_named_files(model_root, names)
    evidence.update(
        {
            "format": "local_hf_assets_v1",
            "config_files": config_files,
            "tokenizer_files": tokenizer_files,
            "weight_files": weight_files,
            "weight_index_files": weight_index_files,
        }
    )
    return evidence


def _trainer_state_provenance(trainer: Any) -> dict[str, Any]:
    """Summarize best-checkpoint selection and final Trainer state."""

    state = trainer.state
    best_checkpoint = state.best_model_checkpoint
    best_metric = state.best_metric
    metric_name = str(trainer.args.metric_for_best_model)
    if not metric_name.startswith("eval_"):
        metric_name = f"eval_{metric_name}"

    checkpoint_step: int | None = None
    if best_checkpoint:
        checkpoint_leaf = Path(str(best_checkpoint)).name
        prefix = "checkpoint-"
        if checkpoint_leaf.startswith(prefix):
            try:
                checkpoint_step = int(checkpoint_leaf[len(prefix) :])
            except ValueError:
                checkpoint_step = None

    eval_entries = [
        entry for entry in state.log_history if metric_name in entry and "step" in entry
    ]
    matching_entries = (
        [entry for entry in eval_entries if int(entry["step"]) == checkpoint_step]
        if checkpoint_step is not None
        else []
    )
    if best_metric is not None:
        metric_matches = [
            entry
            for entry in eval_entries
            if math.isclose(
                float(entry[metric_name]),
                float(best_metric),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]
        if matching_entries:
            exact_matches = [entry for entry in matching_entries if entry in metric_matches]
            if exact_matches:
                matching_entries = exact_matches
        elif metric_matches:
            matching_entries = metric_matches
    best_eval_entry = matching_entries[-1] if matching_entries else None
    best_eval_log = None
    if best_eval_entry is not None:
        best_eval_log = {
            "metric_name": metric_name,
            "metric_value": best_eval_entry[metric_name],
            "epoch": best_eval_entry.get("epoch"),
            "step": best_eval_entry["step"],
        }

    restored_checkpoint = getattr(trainer, "_restored_best_model_checkpoint", None)
    restoration_completed = bool(getattr(trainer, "_best_model_restoration_completed", False))
    restored_best = bool(
        trainer.args.load_best_model_at_end
        and restoration_completed
        and best_checkpoint
        and str(restored_checkpoint) == str(best_checkpoint)
    )
    return {
        "best_model_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        "best_eval_log": best_eval_log,
        "final_global_step": state.global_step,
        "final_epoch": state.epoch,
        "load_best_model_at_end": bool(trainer.args.load_best_model_at_end),
        "restored_best_model_checkpoint": restored_checkpoint,
        "load_best_model_at_end_restored_best_checkpoint": restored_best,
    }


def _training_arguments_provenance(training_args: Any) -> dict[str, Any]:
    """Select manifest-safe values from the constructed TrainingArguments."""

    def enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    checkpointing_kwargs = training_args.gradient_checkpointing_kwargs or {}
    return {
        "optimizer": enum_value(training_args.optim),
        "lr_scheduler_type": enum_value(training_args.lr_scheduler_type),
        "max_grad_norm": float(training_args.max_grad_norm),
        "bf16": bool(training_args.bf16),
        "tf32": bool(training_args.tf32),
        "gradient_checkpointing": bool(training_args.gradient_checkpointing),
        "gradient_checkpointing_use_reentrant": checkpointing_kwargs.get("use_reentrant"),
        "full_determinism": bool(training_args.full_determinism),
        "eval_strategy": enum_value(training_args.eval_strategy),
        "save_strategy": enum_value(training_args.save_strategy),
        "per_device_eval_batch_size": int(training_args.per_device_eval_batch_size),
    }


def _resume_provenance(
    args: argparse.Namespace,
    training_args: Any,
    *,
    base_model_provenance: dict[str, Any],
    train_sha256: str,
    eval_sha256: str,
    dataset_report: dict[str, Any] | None,
    model_lock: dict[str, Any],
    trainer_code_sha256: dict[str, str],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity required to continue a saved checkpoint."""

    datasets = {
        "train_sha256": train_sha256,
        "eval_sha256": eval_sha256,
    }
    if dataset_report is not None:
        datasets["report"] = dataset_report
    return {
        "format": RESUME_PROVENANCE_FORMAT,
        "base_model_provenance": base_model_provenance,
        "model_lock": model_lock,
        "trainer_code_sha256": trainer_code_sha256,
        "datasets": datasets,
        "contract": contract,
        "training": {
            "seed": args.seed,
            "loss": {
                "mode": args.loss_mode,
                "causal_shift": True,
                "ignore_index": -100,
                "token_reduction": "weighted_mean_per_example",
                "example_reduction": "sample_weighted_mean",
                "first_supervised_token_weight": args.first_supervised_token_weight,
            },
            "lora": {
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "target_modules": list(LORA_TARGETS),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "optimization": {
                "learning_rate": args.learning_rate,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_length": args.max_length,
                "prompt_profile": args.prompt_profile,
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": int(training_args.warmup_steps),
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.early_stopping_patience,
                **_training_arguments_provenance(training_args),
            },
        },
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance_mismatch_paths(
    expected: Any,
    observed: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return value-free field paths that differ between provenance records."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [prefix or "<root>"]
        mismatches: list[str] = []
        for key in sorted(expected):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in observed:
                mismatches.append(path)
                continue
            mismatches.extend(
                _provenance_mismatch_paths(
                    expected[key],
                    observed[key],
                    prefix=path,
                )
            )
        if set(observed) - set(expected):
            mismatches.append(f"{prefix}.<unexpected>" if prefix else "<unexpected>")
        return mismatches
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return [prefix or "<root>"]
        mismatches = []
        for index, (expected_item, observed_item) in enumerate(zip(expected, observed)):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            mismatches.extend(
                _provenance_mismatch_paths(
                    expected_item,
                    observed_item,
                    prefix=path,
                )
            )
        return mismatches
    return [] if expected == observed and type(expected) is type(observed) else [prefix or "<root>"]


def _mismatch_summary(mismatches: list[str]) -> str:
    summary = ", ".join(mismatches[:12])
    if len(mismatches) > 12:
        summary += f", ... ({len(mismatches)} fields total)"
    return summary


def _safe_checkpoint_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise RuntimeError("checkpoint artifact has an invalid relative filename")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(
            part in {"", ".", ".."} or _CHECKPOINT_PATH_COMPONENT_RE.fullmatch(part) is None
            for part in relative.parts
        )
    ):
        raise RuntimeError("checkpoint artifact has an invalid relative filename")
    return relative


def _resolve_checkpoint_root(checkpoint: str | Path) -> Path:
    candidate = Path(checkpoint).expanduser()
    if candidate.is_symlink():
        raise RuntimeError("checkpoint directory must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise RuntimeError("checkpoint directory does not exist") from error
    if not stat.S_ISDIR(mode):
        raise RuntimeError("checkpoint path is not a directory")
    if _CHECKPOINT_DIRECTORY_RE.fullmatch(resolved.name) is None:
        raise RuntimeError("checkpoint directory name is invalid")
    return resolved


def _checkpoint_tree_entries(root: Path) -> list[tuple[str, Path, tuple[int, int, int, int, int]]]:
    entries: list[tuple[str, Path, tuple[int, int, int, int, int]]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise RuntimeError("checkpoint artifact tree could not be read") from error
        for child in children:
            path = Path(child.path)
            relative_text = path.relative_to(root).as_posix()
            _safe_checkpoint_relative_path(relative_text)
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise RuntimeError("checkpoint artifact entry could not be read") from error
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError("checkpoint artifact tree contains a symlink")
            if stat.S_ISDIR(mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError("checkpoint artifact tree contains a non-regular file")
            if relative_text == RESUME_PROVENANCE_FILENAME:
                continue
            entries.append((relative_text, path, _stable_stat_identity(metadata)))
    return sorted(entries)


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _checkpoint_file_evidence(path: Path) -> dict[str, Any]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("checkpoint artifact entry is not a regular file")
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or _stable_stat_identity(
                opened
            ) != _stable_stat_identity(before):
                raise RuntimeError("checkpoint artifact changed while hashing")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = path.lstat()
    except OSError as error:
        raise RuntimeError("checkpoint artifact changed while hashing") from error
    if (
        total != before.st_size
        or _stable_stat_identity(after) != _stable_stat_identity(before)
        or _stable_stat_identity(current) != _stable_stat_identity(before)
    ):
        raise RuntimeError("checkpoint artifact changed while hashing")
    return {"bytes": total, "sha256": digest.hexdigest()}


def _require_checkpoint_layout(files: dict[str, dict[str, Any]]) -> None:
    names = set(files)
    if not _REQUIRED_CHECKPOINT_FILES.issubset(names):
        raise RuntimeError("checkpoint artifact set is incomplete")
    if not any("/" not in name and _CHECKPOINT_WEIGHT_RE.fullmatch(name) for name in names):
        raise RuntimeError("checkpoint artifact set has no model weights")
    if not any("/" not in name and _CHECKPOINT_RNG_RE.fullmatch(name) for name in names):
        raise RuntimeError("checkpoint artifact set has no RNG state")


def _checkpoint_artifact_evidence(checkpoint: str | Path) -> dict[str, Any]:
    """Fingerprint the deterministic recursive artifact set of one checkpoint."""

    root = _resolve_checkpoint_root(checkpoint)
    before_entries = _checkpoint_tree_entries(root)
    files = {
        relative: _checkpoint_file_evidence(path) for relative, path, _identity in before_entries
    }
    after_entries = _checkpoint_tree_entries(root)
    before_snapshot = [(name, identity) for name, _path, identity in before_entries]
    after_snapshot = [(name, identity) for name, _path, identity in after_entries]
    if before_snapshot != after_snapshot:
        raise RuntimeError("checkpoint artifact set changed while hashing")
    if not files or sum(item["bytes"] for item in files.values()) <= 0:
        raise RuntimeError("checkpoint artifact set is empty")
    _require_checkpoint_layout(files)
    identity_payload = {
        "format": CHECKPOINT_ARTIFACT_FORMAT,
        "files": files,
        "file_count": len(files),
        "bytes": sum(item["bytes"] for item in files.values()),
    }
    return {
        **identity_payload,
        "identity_sha256": _canonical_json_sha256(identity_payload),
    }


def _validate_checkpoint_artifact_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "format",
        "files",
        "file_count",
        "bytes",
        "identity_sha256",
    }:
        raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")
    if value.get("format") != CHECKPOINT_ARTIFACT_FORMAT:
        raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")
    raw_files = value.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")
    if any(not isinstance(name, str) for name in raw_files):
        raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")

    files: dict[str, dict[str, Any]] = {}
    for name in sorted(raw_files):
        _safe_checkpoint_relative_path(name)
        raw_item = raw_files[name]
        if not isinstance(raw_item, dict) or set(raw_item) != {"bytes", "sha256"}:
            raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")
        size = raw_item.get("bytes")
        digest = raw_item.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")
        files[name] = {"bytes": size, "sha256": digest}

    _require_checkpoint_layout(files)
    identity_payload = {
        "format": CHECKPOINT_ARTIFACT_FORMAT,
        "files": files,
        "file_count": len(files),
        "bytes": sum(item["bytes"] for item in files.values()),
    }
    if (
        isinstance(value.get("file_count"), bool)
        or isinstance(value.get("bytes"), bool)
        or identity_payload["bytes"] <= 0
    ):
        raise RuntimeError("resume provenance has invalid checkpoint artifact evidence")
    if (
        value.get("file_count") != identity_payload["file_count"]
        or value.get("bytes") != identity_payload["bytes"]
        or value.get("identity_sha256") != _canonical_json_sha256(identity_payload)
    ):
        raise RuntimeError("resume provenance has inconsistent checkpoint artifact evidence")
    return {**identity_payload, "identity_sha256": value["identity_sha256"]}


def _read_resume_provenance(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid resume provenance metadata") from error
    if not isinstance(payload, dict):
        raise RuntimeError("invalid resume provenance metadata")
    return payload


def _checkpoint_resume_binding(
    provenance: dict[str, Any],
    artifact: dict[str, Any],
) -> str:
    return _canonical_json_sha256(
        {
            "format": CHECKPOINT_RESUME_BINDING_FORMAT,
            "resume_identity": provenance,
            "checkpoint_artifact": artifact,
        }
    )


def _checkpoint_resume_record(
    provenance: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    if provenance.get("format") != RESUME_PROVENANCE_FORMAT or "checkpoint_artifact" in provenance:
        raise RuntimeError("invalid immutable resume identity")
    return {
        **provenance,
        "checkpoint_artifact": artifact,
        "checkpoint_artifact_binding_sha256": _checkpoint_resume_binding(provenance, artifact),
    }


def _validate_checkpoint_resume_record(
    observed: dict[str, Any],
    expected: dict[str, Any],
    actual_artifact: dict[str, Any],
) -> None:
    expected_keys = set(expected) | {"checkpoint_artifact", "checkpoint_artifact_binding_sha256"}
    if set(observed) != expected_keys:
        raise RuntimeError("resume checkpoint provenance has unexpected fields")
    observed_identity = {key: observed[key] for key in expected}
    mismatches = _provenance_mismatch_paths(expected, observed_identity)
    if mismatches:
        raise RuntimeError(
            "resume checkpoint provenance does not match the current run; "
            f"mismatched fields: {_mismatch_summary(mismatches)}"
        )
    recorded_artifact = _validate_checkpoint_artifact_evidence(observed["checkpoint_artifact"])
    recorded_binding = observed["checkpoint_artifact_binding_sha256"]
    if (
        not isinstance(recorded_binding, str)
        or len(recorded_binding) != 64
        or any(character not in "0123456789abcdef" for character in recorded_binding)
        or recorded_binding != _checkpoint_resume_binding(expected, recorded_artifact)
    ):
        raise RuntimeError("resume checkpoint artifact binding is invalid")
    if recorded_artifact != actual_artifact:
        raise RuntimeError("resume checkpoint artifact bytes do not match provenance")


def _write_resume_provenance(
    directory: str | Path,
    provenance: dict[str, Any],
) -> Path:
    """Atomically bind immutable run identity to one complete checkpoint."""

    root = _resolve_checkpoint_root(directory)
    artifact = _checkpoint_artifact_evidence(root)
    record = _checkpoint_resume_record(provenance, artifact)
    destination = root / RESUME_PROVENANCE_FILENAME
    if os.path.lexists(destination):
        try:
            mode = destination.lstat().st_mode
        except OSError as error:
            raise RuntimeError("resume provenance sidecar could not be read") from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RuntimeError("resume provenance sidecar is not a regular file")
        observed = _read_resume_provenance(destination)
        _validate_checkpoint_resume_record(observed, provenance, artifact)
        return destination

    temporary = root / f".{RESUME_PROVENANCE_FILENAME}.{os.getpid()}.tmp"
    if os.path.lexists(temporary):
        raise RuntimeError("resume provenance temporary path already exists")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()

    if _checkpoint_artifact_evidence(root) != artifact:
        raise RuntimeError("checkpoint artifact changed while writing provenance")
    return destination


def _validate_resume_checkpoint_provenance(
    checkpoint: str | Path,
    expected: dict[str, Any],
) -> Path:
    """Fail closed unless a checkpoint carries exact identity and artifact bytes."""

    resolved = _resolve_checkpoint_root(checkpoint)
    actual_artifact = _checkpoint_artifact_evidence(resolved)
    metadata_path = resolved / RESUME_PROVENANCE_FILENAME
    if not os.path.lexists(metadata_path):
        raise RuntimeError("resume checkpoint is missing required provenance metadata")
    try:
        mode = metadata_path.lstat().st_mode
    except OSError as error:
        raise RuntimeError("resume provenance sidecar could not be read") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError("resume provenance sidecar is not a regular file")
    observed = _read_resume_provenance(metadata_path)
    _validate_checkpoint_resume_record(observed, expected, actual_artifact)
    return resolved


def _prepare_output_directory(
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | None,
) -> tuple[Path, Path | None]:
    """Create a fresh output or validate a controlled in-place resume layout."""

    candidate = Path(output_dir).expanduser()
    if candidate.is_symlink():
        raise RuntimeError("training output directory must not be a symlink")
    if os.path.lexists(candidate):
        try:
            mode = candidate.lstat().st_mode
        except OSError as error:
            raise RuntimeError("training output directory could not be read") from error
        if not stat.S_ISDIR(mode):
            raise RuntimeError("training output path is not a directory")
        root = candidate.resolve(strict=True)
    else:
        if resume_from_checkpoint is not None:
            raise RuntimeError("resume requires an existing output directory")
        candidate.mkdir(parents=True, exist_ok=False)
        root = candidate.resolve(strict=True)

    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise RuntimeError("training output directory could not be read") from error
    if resume_from_checkpoint is None:
        if entries:
            raise RuntimeError(
                "refusing to use a preexisting nonempty output directory without "
                "--resume-from-checkpoint"
            )
        return root, None

    checkpoint = _resolve_checkpoint_root(resume_from_checkpoint)
    if checkpoint.parent != root:
        raise RuntimeError("resume checkpoint must be a direct child of the output directory")
    if not entries:
        raise RuntimeError("resume output directory is empty")
    for entry in entries:
        try:
            mode = entry.lstat().st_mode
        except OSError as error:
            raise RuntimeError("resume output directory entry could not be read") from error
        if (
            stat.S_ISLNK(mode)
            or not stat.S_ISDIR(mode)
            or _CHECKPOINT_DIRECTORY_RE.fullmatch(entry.name) is None
        ):
            raise RuntimeError("resume output directory has a stale or unexpected layout")
    return root, checkpoint


def _create_adapter_destination(output_dir: str | Path) -> Path:
    """Atomically reserve a new adapter directory without overwriting stale output."""

    root_candidate = Path(output_dir).expanduser()
    if root_candidate.is_symlink():
        raise RuntimeError("training output directory must not be a symlink")
    try:
        root = root_candidate.resolve(strict=True)
        mode = root.lstat().st_mode
    except OSError as error:
        raise RuntimeError("training output directory could not be read") from error
    if not stat.S_ISDIR(mode):
        raise RuntimeError("training output path is not a directory")
    for name in ("adapter", "run_manifest.json"):
        if os.path.lexists(root / name):
            raise RuntimeError("training output has a stale adapter or manifest layout")
    adapter = root / "adapter"
    try:
        adapter.mkdir(exist_ok=False)
    except OSError as error:
        raise RuntimeError("adapter destination could not be reserved") from error
    return adapter.resolve(strict=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _prompt_profile(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    try:
        return PROMPT_PROFILE_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(PROMPT_PROFILE_ALIASES))
        raise argparse.ArgumentTypeError(
            f"unknown prompt profile {value!r}; choose one of: {choices}"
        ) from error


def _load_optional_contract(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return None


def _android_messages(sender: str, sms: str) -> list[dict[str, str]]:
    module = _load_optional_contract("lfm25.android_contract")
    if module is not None:
        helper = getattr(module, "android_extraction_messages", None)
        if not callable(helper):
            raise RuntimeError("lfm25.android_contract lacks android_extraction_messages")
        messages = helper(sender, sms)
    else:
        try:
            from DATA.utils import doc_to_text
        except ImportError as error:
            raise RuntimeError(
                "the Android prompt profile requires lfm25.android_contract or DATA.utils"
            ) from error
        messages = [
            {"role": "system", "content": ANDROID_OUTER_SYSTEM_PROMPT},
            {"role": "user", "content": doc_to_text({"sender": sender, "sms": sms})},
        ]
    return _validate_prompt_messages(messages, contract="Android")


def _selector_messages(sender: str, sms: str) -> list[dict[str, str]]:
    module = _load_optional_contract("lfm25.candidates")
    if module is None:
        raise RuntimeError("the candidate-selector profile requires lfm25.candidates")
    helper = getattr(module, "candidate_selector_messages", None)
    if not callable(helper):
        raise RuntimeError("lfm25.candidates lacks candidate_selector_messages")
    return _validate_prompt_messages(helper(sender, sms), contract="candidate-selector")


def _validate_prompt_messages(messages: Any, *, contract: str) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) != 2:
        raise RuntimeError(f"the {contract} contract must produce exactly system and user messages")
    prepared = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in messages
        if isinstance(item, dict)
    ]
    if len(prepared) != 2 or [item["role"] for item in prepared] != ["system", "user"]:
        raise RuntimeError(f"the {contract} contract must produce exactly system and user messages")
    return prepared


def _contract_provenance(prompt_profile: str) -> dict[str, Any]:
    if prompt_profile == "legacy":
        return {
            "profile": "legacy",
            "contract": "lfm25_short_extraction",
            "contract_version": 1,
            "prompt_sha256": hashlib.sha256(PRODUCTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        }
    if prompt_profile == "candidate_selector":
        module = _load_optional_contract("lfm25.candidates")
        if module is None:
            raise RuntimeError("the candidate-selector profile requires lfm25.candidates")
        helper = getattr(module, "contract_provenance", None)
        provenance = helper() if callable(helper) else {}
        if not isinstance(provenance, dict):
            raise RuntimeError("lfm25.candidates.contract_provenance() must return a dict")
        selector_prompt = str(getattr(module, "SELECTOR_SYSTEM_PROMPT", ""))
        source_path = Path(str(module.__file__))
        return {
            "profile": "candidate_selector",
            "contract": "grounded_candidate_selector",
            "prompt_sha256": hashlib.sha256(selector_prompt.encode("utf-8")).hexdigest(),
            "source_sha256": _sha256(source_path),
            **provenance,
        }
    module = _load_optional_contract("lfm25.android_contract")
    if module is not None:
        helper = getattr(module, "contract_provenance", None)
        if not callable(helper):
            raise RuntimeError("lfm25.android_contract lacks contract_provenance")
        provenance = helper()
        if not isinstance(provenance, dict):
            raise RuntimeError("lfm25.android_contract.contract_provenance() must return a dict")
        return {"profile": "android", **provenance}
    source = REPO_ROOT / "DATA" / "utils.py"
    return {
        "profile": "android",
        "contract": "pocketfinancer_android",
        "contract_version": "DATA.utils fallback",
        "prompt_source": "DATA.utils.doc_to_text",
        "prompt_source_sha256": _sha256(source) if source.is_file() else None,
        "outer_system_prompt_sha256": hashlib.sha256(
            ANDROID_OUTER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }


def _assistant_text(value: Any) -> str:
    gold = parse_gold(value)
    if gold is None:
        return "null"
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":"))


def _row_target(row: dict[str, Any]) -> tuple[Any, bool]:
    for key in ("expected", "label"):
        if key in row:
            return row[key], True
    return None, False


def _existing_assistant_text(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    if (
        not messages
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "assistant"
    ):
        raise ValueError("every SFT row must end with an assistant message")
    return str(messages[-1].get("content", ""))


def _selector_assistant_text(row: dict[str, Any], sms: str) -> str:
    module = _load_optional_contract("lfm25.candidates")
    if module is None:
        raise RuntimeError("the candidate-selector profile requires lfm25.candidates")
    target, has_target = _row_target(row)
    candidates = module.extract_candidates(sms)
    if has_target:
        selection = module.selector_target(target, candidates)
    else:
        existing = _existing_assistant_text(row)
        if existing is None:
            raise ValueError(
                "candidate-selector rows require expected, label, or assistant content"
            )
        try:
            selection = json.loads(existing)
            module.reconstruct_transaction(selection, candidates)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                "assistant completion violates the candidate-selector contract"
            ) from error
    return json.dumps(selection, ensure_ascii=False, separators=(",", ":"))


def _messages(row: dict[str, Any], prompt_profile: str = "legacy") -> list[dict[str, str]]:
    prompt_profile = _prompt_profile(prompt_profile)
    sender = str(row.get("sender", ""))
    sms = str(row.get("sms", ""))
    if prompt_profile == "legacy" and isinstance(row.get("messages"), list):
        messages = row["messages"]
    elif prompt_profile == "candidate_selector":
        messages = _selector_messages(sender, sms)
        messages.append({"role": "assistant", "content": _selector_assistant_text(row, sms)})
    else:
        target, has_target = _row_target(row)
        existing = _existing_assistant_text(row)
        if not has_target and existing is None:
            raise ValueError("SFT rows require expected, label, or assistant content")
        assistant = _assistant_text(target) if has_target else str(existing)
        messages = (
            _android_messages(sender, sms)
            if prompt_profile == "android"
            else extraction_messages(sender, sms)
        )
        messages.append({"role": "assistant", "content": assistant})
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("every SFT row must end with an assistant message")
    if prompt_profile == "candidate_selector":
        _selector_assistant_text({**row, "messages": messages}, sms)
    else:
        parsed = parse_prediction(str(messages[-1].get("content", "")))
        if parsed.status == "invalid":
            raise ValueError("assistant completion violates the strict extraction contract")
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


class OverlengthDatasetError(ValueError):
    def __init__(self, path: Path, max_length: int, rows: list[tuple[int, int]]):
        self.path = path
        self.max_length = max_length
        self.overlength_count = len(rows)
        self.max_observed_length = max(length for _, length in rows)
        preview = ", ".join(f"{index}:{length}" for index, length in rows[:8])
        if len(rows) > 8:
            preview += ", ..."
        super().__init__(
            f"{len(rows)} tokenized row(s) in {path} exceed --max-length {max_length}; "
            f"maximum observed length is {self.max_observed_length}; row:length {preview}. "
            "Raise the limit instead of truncating."
        )


class CompletionDataset:
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        prompt_profile: str = "legacy",
    ):
        if max_length <= 1:
            raise ValueError("max_length must be greater than one")
        self.path = path
        self.prompt_profile = _prompt_profile(prompt_profile)
        self.features: list[dict[str, Any]] = []
        lengths: list[int] = []
        supervised_lengths: list[int] = []
        sample_weights: list[float] = []
        explicit_sample_weights = 0
        overlength: list[tuple[int, int]] = []
        for row_index, row in enumerate(_read_jsonl(path), start=1):
            raw_weight = row.get("sample_weight", 1.0)
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError(f"row {row_index} sample_weight must be numeric")
            sample_weight = float(raw_weight)
            if not math.isfinite(sample_weight) or sample_weight <= 0:
                raise ValueError(f"row {row_index} sample_weight must be finite and positive")
            explicit_sample_weights += int("sample_weight" in row)

            messages = _messages(row, self.prompt_profile)
            prompt_encoding = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            full_encoding = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
            )
            prompt_ids = _token_ids(prompt_encoding)
            input_ids = _token_ids(full_encoding)
            common = 0
            for left, right in zip(prompt_ids, input_ids):
                if left != right:
                    break
                common += 1
            if common == 0:
                raise ValueError(
                    f"row {row_index} chat template has no stable prompt/completion boundary"
                )
            if common == len(input_ids):
                raise ValueError(
                    f"row {row_index} chat template left no assistant completion tokens"
                )
            if len(input_ids) > max_length:
                overlength.append((row_index, len(input_ids)))
                continue
            labels = [-100] * common + input_ids[common:]
            supervised_length = sum(label != -100 for label in labels[1:])
            if supervised_length == 0:
                raise ValueError(f"row {row_index} has no causally supervised completion tokens")
            self.features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "sample_weight": sample_weight,
                }
            )
            lengths.append(len(input_ids))
            supervised_lengths.append(supervised_length)
            sample_weights.append(sample_weight)
        if overlength:
            raise OverlengthDatasetError(path, max_length, overlength)
        if not self.features:
            raise ValueError(f"no rows in {path}")
        ordered = sorted(lengths)
        p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        self.stats = {
            "rows": len(self.features),
            "tokens_min": min(lengths),
            "tokens_mean": round(statistics.fmean(lengths), 2),
            "tokens_p95": ordered[p95_index],
            "tokens_max": max(lengths),
            "completion_tokens_mean": round(statistics.fmean(supervised_lengths), 2),
            "completion_tokens_min": min(supervised_lengths),
            "max_length_limit": max_length,
            "overlength_rows": 0,
            "prompt_profile": self.prompt_profile,
            "sample_weight_field": "sample_weight",
            "sample_weight_explicit_rows": explicit_sample_weights,
            "sample_weight_min": min(sample_weights),
            "sample_weight_mean": round(statistics.fmean(sample_weights), 6),
            "sample_weight_max": max(sample_weights),
            "sample_weight_sum": round(sum(sample_weights), 6),
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def _token_ids(encoding: Any) -> list[int]:
    values = (
        encoding["input_ids"] if hasattr(encoding, "keys") and "input_ids" in encoding else encoding
    )
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("chat template unexpectedly returned a batch")
        values = values[0]
    return [int(value) for value in values]


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer has no pad token")

    def __call__(self, features):
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_masks, labels, sample_weights = [], [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            sample_weights.append(float(feature.get("sample_weight", 1.0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TRAINING_ARTIFACTS/base/LFM2.5-350M")
    parser.add_argument("--model-lock", type=Path, default=MODEL_LOCK_PATH)
    parser.add_argument("--dataset-report", type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-profile",
        "--contract",
        dest="prompt_profile",
        type=_prompt_profile,
        default="pocketfinancer",
        help=(
            "prompt profile: pocketfinancer (default); legacy and "
            "candidate_selector are historical research profiles"
        ),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="defaults to 3072 for Android and 512 for legacy/selector profiles",
    )
    parser.add_argument("--loss-mode", choices=[LOSS_MODE], default=LOSS_MODE)
    parser.add_argument(
        "--first-supervised-token-weight",
        "--first-token-weight",
        "--decision-token-weight",
        dest="first_supervised_token_weight",
        type=float,
        default=3.0,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    expected_train = "candidate_protocol_v1_train.jsonl"
    expected_eval = "candidate_protocol_v1_dev.jsonl"
    observes_candidate_name = args.train.name == expected_train or args.eval.name == expected_eval
    candidate_bundle = args.train.name == expected_train and args.eval.name == expected_eval
    if observes_candidate_name and not candidate_bundle:
        parser.error("Candidate V1 training requires its matched train/dev filenames")
    if candidate_bundle and args.dataset_report is None:
        parser.error("--dataset-report is required for Candidate V1 training")
    if not candidate_bundle and args.dataset_report is not None:
        parser.error("--dataset-report requires the Candidate V1 train/dev bundle")
    if (
        args.dataset_report is not None
        and args.dataset_report.name != "candidate_protocol_v1_report.json"
    ):
        parser.error("--dataset-report must name candidate_protocol_v1_report.json")
    if args.model_lock.name != "model.lock.json":
        parser.error("--model-lock must name model.lock.json")
    if args.max_length is None:
        args.max_length = (
            ANDROID_CONTEXT_LENGTH if args.prompt_profile == "android" else LEGACY_MAX_LENGTH
        )
    if args.max_length <= 1:
        parser.error("--max-length must be greater than one")
    if args.prompt_profile == "android" and args.max_length > ANDROID_CONTEXT_LENGTH:
        parser.error(
            f"Android contract context is {ANDROID_CONTEXT_LENGTH}; --max-length cannot exceed it"
        )
    if not math.isfinite(args.first_supervised_token_weight) or (
        args.first_supervised_token_weight <= 0
    ):
        parser.error("--first-supervised-token-weight must be finite and greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir, resume_checkpoint = _prepare_output_directory(
        args.output_dir,
        args.resume_from_checkpoint,
    )
    args.output_dir = output_dir
    args.resume_from_checkpoint = str(resume_checkpoint) if resume_checkpoint is not None else None
    run_started = time.perf_counter()

    os.environ.setdefault("WANDB_DISABLED", "true")

    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    import peft
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    class CompletionNormalizedTrainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self._best_model_restoration_completed = False
            self._restored_best_model_checkpoint = None

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            sample_weight = model_inputs.pop("sample_weight", None)
            outputs = model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = normalized_completion_cross_entropy(
                logits,
                labels,
                sample_weight=sample_weight,
                first_supervised_token_weight=args.first_supervised_token_weight,
            )
            return (loss, outputs) if return_outputs else loss

        def _load_best_model(self):
            super()._load_best_model()
            self._best_model_restoration_completed = True
            self._restored_best_model_checkpoint = self.state.best_model_checkpoint

    class ResumeProvenanceCallback(TrainerCallback):
        def __init__(self, provenance: dict[str, Any]):
            self._provenance = provenance

        def on_save(self, training_args, state, control, **kwargs):
            del kwargs
            checkpoint_dir = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            _write_resume_provenance(checkpoint_dir, self._provenance)
            return control

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BF16 LoRA training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    train_sha256 = _sha256(args.train)
    eval_sha256 = _sha256(args.eval)
    dataset_report = _candidate_dataset_report_provenance(
        args.train, args.eval, args.dataset_report
    )
    training_arm = (
        "selector"
        if args.prompt_profile in {"candidate_protocol_v1", "candidate_selector"}
        else "direct"
    )
    trainer_code_sha256 = trainer_code_fingerprints(REPO_ROOT, training_arm)
    base_model_provenance = _local_hf_model_provenance(args.model)
    model_lock = _locked_model_provenance(base_model_provenance, args.model_lock)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    contract = _contract_provenance(args.prompt_profile)
    train_dataset = CompletionDataset(
        args.train,
        tokenizer,
        args.max_length,
        prompt_profile=args.prompt_profile,
    )
    eval_dataset = CompletionDataset(
        args.eval,
        tokenizer,
        args.max_length,
        prompt_profile=args.prompt_profile,
    )
    print(
        json.dumps(
            {
                "dataset_preflight": {
                    "contract": args.prompt_profile,
                    "max_length": args.max_length,
                    "train": train_dataset.stats,
                    "eval": eval_dataset.stats,
                }
            },
            sort_keys=True,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    target_coverage = target_module_inventory(model)
    print(json.dumps({"lora_target_preflight": target_coverage}, sort_keys=True))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=LORA_TARGETS,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise RuntimeError("LoRA attached no trainable parameters")
    model.print_trainable_parameters()

    batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / args.gradient_accumulation))
    total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = round(total_updates * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cache=False,
    )
    resume_provenance = _resume_provenance(
        args,
        training_args,
        base_model_provenance=base_model_provenance,
        train_sha256=train_sha256,
        eval_sha256=eval_sha256,
        dataset_report=dataset_report,
        model_lock=model_lock,
        trainer_code_sha256=trainer_code_sha256,
        contract=contract,
    )
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = str(
            _validate_resume_checkpoint_provenance(
                args.resume_from_checkpoint,
                resume_provenance,
            )
        )
    trainer = CompletionNormalizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            ResumeProvenanceCallback(resume_provenance),
        ],
    )
    training_started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    training_wall_time_seconds = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    eval_metrics = trainer.evaluate()
    evaluation_wall_time_seconds = time.perf_counter() - evaluation_started
    trainer_state = _trainer_state_provenance(trainer)

    adapter_dir = _create_adapter_destination(args.output_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    _require_unchanged("training dataset", train_sha256, _sha256(args.train))
    _require_unchanged("evaluation dataset", eval_sha256, _sha256(args.eval))
    _require_unchanged(
        "candidate dataset report",
        dataset_report,
        _candidate_dataset_report_provenance(args.train, args.eval, args.dataset_report),
    )
    current_base_model = _local_hf_model_provenance(args.model)
    _require_unchanged("base model", base_model_provenance, current_base_model)
    _require_unchanged(
        "model lock",
        model_lock,
        _locked_model_provenance(current_base_model, args.model_lock),
    )
    _require_unchanged(
        "trainer code",
        trainer_code_sha256,
        trainer_code_fingerprints(REPO_ROOT, training_arm),
    )
    adapter_artifact = adapter_artifact_evidence(adapter_dir)
    run_wall_time_seconds = time.perf_counter() - run_started

    manifest = {
        "base_model": str(Path(args.model).resolve()),
        "manifest_format": RUN_MANIFEST_FORMAT,
        "base_model_provenance": base_model_provenance,
        "train_file": str(args.train.resolve()),
        "model_lock": model_lock,
        "adapter_artifact": adapter_artifact,
        "trainer_code_sha256": trainer_code_sha256,
        "train_sha256": train_sha256,
        "eval_file": str(args.eval.resolve()),
        "eval_sha256": eval_sha256,
        "train_stats": train_dataset.stats,
        **({"dataset_report": dataset_report} if dataset_report is not None else {}),
        "eval_stats": eval_dataset.stats,
        "contract": contract,
        "loss": {
            "mode": args.loss_mode,
            "causal_shift": True,
            "ignore_index": -100,
            "token_reduction": "weighted_mean_per_example",
            "example_reduction": "sample_weighted_mean",
            "weights": {
                "sample_weight_field": "sample_weight",
                "default_sample_weight": 1.0,
                "first_supervised_token_weight": args.first_supervised_token_weight,
                "train_explicit_sample_weight_rows": train_dataset.stats[
                    "sample_weight_explicit_rows"
                ],
                "eval_explicit_sample_weight_rows": eval_dataset.stats[
                    "sample_weight_explicit_rows"
                ],
            },
        },
        "seed": args.seed,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": LORA_TARGETS,
            "target_module_coverage": target_coverage,
            "trainable_parameters": trainable,
            "total_parameters_with_adapter": total,
            "trainable_percent": round(100 * trainable / total, 6),
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "max_length": args.max_length,
            "prompt_profile": args.prompt_profile,
            "loss_mode": args.loss_mode,
            "first_supervised_token_weight": args.first_supervised_token_weight,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            **_training_arguments_provenance(training_args),
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "trainer_state": trainer_state,
        "timing": {
            "run_wall_time_seconds_through_adapter_save": round(run_wall_time_seconds, 3),
            "training_wall_time_seconds": round(training_wall_time_seconds, 3),
            "final_evaluation_wall_time_seconds": round(evaluation_wall_time_seconds, 3),
        },
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "privacy": {
            "report_to": "none",
            "push_to_hub": False,
            "raw_examples_logged": False,
        },
    }
    manifest["artifact_binding_sha256"] = training_manifest_artifact_binding(manifest)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "adapter_dir": str(adapter_dir),
                "trainable_parameters": trainable,
                "eval_loss": eval_metrics.get("eval_loss"),
                "peak_vram_mib": manifest["peak_vram_mib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
