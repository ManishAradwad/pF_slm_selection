#!/usr/bin/env python3
"""Run and summarize the local LoRA rank/LR and finalist-seed ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from lfm25.prompts import PRODUCTION_SYSTEM_PROMPT  # noqa: E402
from lfm25.provenance import (  # noqa: E402
    code_fingerprints,
    file_sha256,
    fingerprint_file,
    fingerprint_named_files,
)
from lfm25.training_loss import LOSS_MODE  # noqa: E402

LEGACY_PROMPT_PROFILE = "legacy"
DEFAULT_FIRST_SUPERVISED_TOKEN_WEIGHT = 1.0
DEFAULT_MAX_NEW_TOKENS = 96
ADAPTER_FINGERPRINT_FILES = ("adapter_model.safetensors", "adapter_config.json")
QUALITY_KEYS = (
    "json_validity",
    "schema_validity",
    "four_field_exact_match",
    "four_field_exact_match_ci95",
    "transaction_only_exact_match",
    "conditional_ghost_rate",
    "conditional_ghost_rate_ci95",
    "conditional_miss_rate",
    "conditional_miss_rate_ci95",
    "transaction_precision",
    "transaction_recall",
    "transaction_f1",
    "field_accuracy_on_transactions",
)


def _legacy_contract_settings(common: dict[str, Any]) -> tuple[str, str, float]:
    """Resolve explicit trainer settings for this legacy-only experiment runner."""

    prompt_profile = str(common.get("prompt_profile", LEGACY_PROMPT_PROFILE))
    if prompt_profile != LEGACY_PROMPT_PROFILE:
        raise ValueError(
            "run_lfm25_experiments.py only supports prompt_profile='legacy'; "
            "use the contract-specific v2 training/evaluation scripts for other profiles"
        )
    loss_mode = str(common.get("loss_mode", LOSS_MODE))
    if loss_mode != LOSS_MODE:
        raise ValueError(
            f"unsupported loss_mode {loss_mode!r}; expected the trainer mode {LOSS_MODE!r}"
        )
    first_token_weight = float(
        common.get(
            "first_supervised_token_weight",
            DEFAULT_FIRST_SUPERVISED_TOKEN_WEIGHT,
        )
    )
    if not math.isfinite(first_token_weight) or first_token_weight <= 0:
        raise ValueError("first_supervised_token_weight must be finite and greater than zero")
    return prompt_profile, loss_mode, first_token_weight


def _legacy_decode_settings(common: dict[str, Any], *, seed: int) -> dict[str, Any]:
    max_new_tokens = int(common.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    return {
        "engine": "transformers",
        "grammar_constrained": False,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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


def _selection_key(record: dict[str, Any]) -> tuple[float, float, float]:
    metrics = record["dev_metrics"]
    exact = metrics.get("four_field_exact_match")
    ghost = metrics.get("conditional_ghost_rate")
    loss = record.get("eval_loss")
    return (
        -float(exact if exact is not None else -1.0),
        float(ghost if ghost is not None else 1.0),
        float(loss if loss is not None else float("inf")),
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps({"command": command}, sort_keys=True))
    if dry_run:
        return
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def _common_training_flags(common: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key in (
        "alpha",
        "dropout",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "max_length",
        "warmup_ratio",
        "weight_decay",
        "early_stopping_patience",
    ):
        if key not in common:
            raise ValueError(f"missing common experiment setting: {key}")
        flags.extend((f"--{key.replace('_', '-')}", str(common[key])))
    prompt_profile, loss_mode, first_token_weight = _legacy_contract_settings(common)
    flags.extend(
        (
            "--prompt-profile",
            prompt_profile,
            "--loss-mode",
            loss_mode,
            "--first-supervised-token-weight",
            str(first_token_weight),
        )
    )
    return flags


def _validate_training_reuse(
    manifest: dict[str, Any],
    *,
    base_model: Path,
    train: Path,
    dev: Path,
    rank: int,
    learning_rate: float,
    seed: int,
    common: dict[str, Any],
) -> None:
    """Fail closed when an existing run does not match the requested experiment."""

    mismatches: list[str] = []

    def require(name: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            mismatches.append(name)

    require("base_model", Path(str(manifest.get("base_model", ""))).resolve(), base_model)
    require("train_file", Path(str(manifest.get("train_file", ""))).resolve(), train)
    require("eval_file", Path(str(manifest.get("eval_file", ""))).resolve(), dev)
    require("train_sha256", manifest.get("train_sha256"), file_sha256(train))
    require("eval_sha256", manifest.get("eval_sha256"), file_sha256(dev))
    require("seed", manifest.get("seed"), seed)
    lora = manifest.get("lora")
    if not isinstance(lora, dict):
        lora = {}
    optimization = manifest.get("optimization")
    if not isinstance(optimization, dict):
        optimization = {}
    require("rank", lora.get("rank"), rank)
    require("learning_rate", optimization.get("learning_rate"), learning_rate)
    prompt_profile, loss_mode, first_token_weight = _legacy_contract_settings(common)
    expected_optimization = {
        "alpha": (lora, "alpha"),
        "dropout": (lora, "dropout"),
        "epochs": (optimization, "epochs_requested"),
        "batch_size": (optimization, "batch_size"),
        "gradient_accumulation": (optimization, "gradient_accumulation"),
        "max_length": (optimization, "max_length"),
        "warmup_ratio": (optimization, "warmup_ratio"),
        "weight_decay": (optimization, "weight_decay"),
        "early_stopping_patience": (optimization, "early_stopping_patience"),
    }
    for config_name, (container, manifest_name) in expected_optimization.items():
        require(config_name, container.get(manifest_name), common[config_name])
    require(
        "effective_batch_size",
        optimization.get("effective_batch_size"),
        int(common["batch_size"]) * int(common["gradient_accumulation"]),
    )
    require("optimization.prompt_profile", optimization.get("prompt_profile"), prompt_profile)
    require("optimization.loss_mode", optimization.get("loss_mode"), loss_mode)
    require(
        "optimization.first_supervised_token_weight",
        optimization.get("first_supervised_token_weight"),
        first_token_weight,
    )

    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        contract = {}
    expected_contract = {
        "profile": LEGACY_PROMPT_PROFILE,
        "contract": "lfm25_short_extraction",
        "contract_version": 1,
        "prompt_sha256": hashlib.sha256(
            PRODUCTION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    for key, expected in expected_contract.items():
        require(f"contract.{key}", contract.get(key), expected)

    loss = manifest.get("loss")
    if not isinstance(loss, dict):
        loss = {}
    require("loss.mode", loss.get("mode"), loss_mode)
    require("loss.causal_shift", loss.get("causal_shift"), True)
    require("loss.ignore_index", loss.get("ignore_index"), -100)
    require("loss.token_reduction", loss.get("token_reduction"), "weighted_mean_per_example")
    require("loss.example_reduction", loss.get("example_reduction"), "sample_weighted_mean")
    weights = loss.get("weights")
    if not isinstance(weights, dict):
        weights = {}
    require("loss.weights.sample_weight_field", weights.get("sample_weight_field"), "sample_weight")
    require("loss.weights.default_sample_weight", weights.get("default_sample_weight"), 1.0)
    require(
        "loss.weights.first_supervised_token_weight",
        weights.get("first_supervised_token_weight"),
        first_token_weight,
    )
    if mismatches:
        rendered = ", ".join(sorted(mismatches))
        raise RuntimeError(
            f"refusing stale training reuse; mismatched fields: {rendered}. "
            "Use a new artifacts root for a changed experiment."
        )


def _validate_evaluation_reuse(
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    *,
    dataset: Path,
    adapter: Path,
    seed: int,
    common: dict[str, Any],
) -> None:
    """Require evidence that cached legacy evaluation matches all material inputs."""

    _legacy_contract_settings(common)
    mismatches: list[str] = []

    def require(name: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            mismatches.append(name)

    eval_stats = manifest.get("eval_stats")
    if not isinstance(eval_stats, dict):
        eval_stats = {}
    expected_rows = eval_stats.get("rows")
    if not isinstance(expected_rows, int):
        mismatches.append("manifest.eval_stats.rows")
    counts = metrics.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    observed_rows = counts.get("rows", metrics.get("n"))
    require("row_count", observed_rows, expected_rows)
    provenance = metrics.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}

    expected_dataset = fingerprint_file(dataset)
    observed_dataset = provenance.get("dataset")
    if not isinstance(observed_dataset, dict):
        observed_dataset = {}
    require("dataset.sha256", observed_dataset.get("sha256"), expected_dataset["sha256"])
    require("dataset.bytes", observed_dataset.get("bytes"), expected_dataset["bytes"])
    require("manifest.eval_sha256", manifest.get("eval_sha256"), expected_dataset["sha256"])

    expected_adapter = fingerprint_named_files(adapter, ADAPTER_FINGERPRINT_FILES)
    expected_adapter_files = expected_adapter["files"]
    if set(expected_adapter_files) != set(ADAPTER_FINGERPRINT_FILES):
        mismatches.append("adapter.required_files")
    observed_adapter = provenance.get("adapter")
    if not isinstance(observed_adapter, dict):
        observed_adapter = {}
    require("adapter.files", observed_adapter.get("files"), expected_adapter_files)

    # The legacy evaluator did not emit a named contract. Its code fingerprints
    # bind reuse to the short prompt, strict parser, metric implementation, and grammar.
    require("contract.code_sha256", provenance.get("code_sha256"), code_fingerprints(REPO_ROOT))
    expected_decode = _legacy_decode_settings(common, seed=seed)
    observed_decode = provenance.get("decode")
    if not isinstance(observed_decode, dict):
        observed_decode = {}
    for key, expected in expected_decode.items():
        require(f"decode.{key}", observed_decode.get(key), expected)
    require("row_limit", provenance.get("row_limit"), None)
    require("provenance.row_count", provenance.get("row_count"), expected_rows)

    if mismatches:
        rendered = ", ".join(sorted(mismatches))
        raise RuntimeError(
            f"refusing stale evaluation reuse; mismatched fields: {rendered}. "
            "Delete the cached evaluation or use a new results root."
        )


def _run_one(
    *,
    name: str,
    rank: int,
    learning_rate: float,
    seed: int,
    base_model: Path,
    train: Path,
    dev: Path,
    common: dict[str, Any],
    artifacts_root: Path,
    results_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    run_dir = artifacts_root / name
    eval_dir = results_root / name / "dev"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = eval_dir / "metrics.json"
    decode = _legacy_decode_settings(common, seed=seed)

    if not manifest_path.exists():
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_lfm25_lora.py"),
            "--model",
            str(base_model),
            "--train",
            str(train),
            "--eval",
            str(dev),
            "--output-dir",
            str(run_dir),
            "--rank",
            str(rank),
            "--learning-rate",
            str(learning_rate),
            "--seed",
            str(seed),
            *_common_training_flags(common),
        ]
        _run(command, dry_run=dry_run)
    else:
        existing_manifest = _read_json(manifest_path)
        _validate_training_reuse(
            existing_manifest,
            base_model=base_model,
            train=train,
            dev=dev,
            rank=rank,
            learning_rate=learning_rate,
            seed=seed,
            common=common,
        )
        missing_adapter_files = [
            name
            for name in ADAPTER_FINGERPRINT_FILES
            if not (run_dir / "adapter" / name).is_file()
        ]
        if missing_adapter_files:
            rendered = ", ".join(missing_adapter_files)
            raise RuntimeError(f"refusing training reuse; adapter files are missing: {rendered}")
        print(json.dumps({"reuse_training": name}, sort_keys=True))

    if not metrics_path.exists():
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_lfm25_hf.py"),
            "--model",
            str(base_model),
            "--adapter",
            str(run_dir / "adapter"),
            "--dataset",
            str(dev),
            "--output-dir",
            str(eval_dir),
            "--batch-size",
            str(common.get("eval_batch_size", 8)),
            "--max-new-tokens",
            str(decode["max_new_tokens"]),
            "--seed",
            str(seed),
        ]
        _run(command, dry_run=dry_run)
    else:
        _validate_evaluation_reuse(
            _read_json(metrics_path),
            _read_json(manifest_path),
            dataset=dev,
            adapter=run_dir / "adapter",
            seed=seed,
            common=common,
        )
        print(json.dumps({"reuse_evaluation": name}, sort_keys=True))

    if dry_run:
        return {
            "name": name,
            "rank": rank,
            "learning_rate": learning_rate,
            "seed": seed,
            "dry_run": True,
        }
    manifest = _read_json(manifest_path)
    metrics = _read_json(metrics_path)
    return {
        "name": name,
        "rank": rank,
        "learning_rate": learning_rate,
        "seed": seed,
        "artifact_dir": str(run_dir),
        "evaluation_dir": str(eval_dir),
        "eval_loss": manifest.get("eval_metrics", {}).get("eval_loss"),
        "peak_training_vram_mib": manifest.get("peak_vram_mib"),
        "epochs_completed": manifest.get("train_metrics", {}).get("epoch"),
        "dev_metrics": {key: metrics.get(key) for key in QUALITY_KEYS},
        "dev_slices": metrics.get("slices", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "lfm25" / "experiments.json",
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "TRAINING_ARTIFACTS" / "lfm25_experiments",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "RESULTS" / "lfm25" / "experiments",
    )
    parser.add_argument("--comparison-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _read_json(args.config)
    common = config["common"]
    _legacy_contract_settings(common)
    base_model = (REPO_ROOT / config["base_model"]).resolve()
    train = args.train.resolve(strict=not args.dry_run)
    dev = args.dev.resolve(strict=not args.dry_run)
    comparison: list[dict[str, Any]] = []
    for experiment in config["comparison"]:
        comparison.append(
            _run_one(
                name=str(experiment["name"]),
                rank=int(experiment["rank"]),
                learning_rate=float(experiment["learning_rate"]),
                seed=int(experiment["seed"]),
                base_model=base_model,
                train=train,
                dev=dev,
                common=common,
                artifacts_root=args.artifacts_root,
                results_root=args.results_root,
                dry_run=args.dry_run,
            )
        )
    if args.dry_run:
        return 0

    hyperparameter_winner = min(comparison, key=_selection_key)
    finalists = [hyperparameter_winner]
    if not args.comparison_only:
        for seed in config["finalist_seeds"]:
            seed = int(seed)
            if seed == hyperparameter_winner["seed"]:
                continue
            name = (
                f"final_r{hyperparameter_winner['rank']}_"
                f"lr{str(hyperparameter_winner['learning_rate']).replace('.', 'p')}_"
                f"seed{seed}"
            )
            finalists.append(
                _run_one(
                    name=name,
                    rank=int(hyperparameter_winner["rank"]),
                    learning_rate=float(hyperparameter_winner["learning_rate"]),
                    seed=seed,
                    base_model=base_model,
                    train=train,
                    dev=dev,
                    common=common,
                    artifacts_root=args.artifacts_root,
                    results_root=args.results_root,
                    dry_run=False,
                )
            )
    selected_run = min(finalists, key=_selection_key)
    summary = {
        "selection_rule": config["selection_rule"],
        "train": str(train),
        "dev": str(dev),
        "comparison": comparison,
        "hyperparameter_winner": hyperparameter_winner["name"],
        "finalist_seed_runs": finalists,
        "selected_run": selected_run["name"],
        "selected_adapter": str(Path(selected_run["artifact_dir"]) / "adapter"),
    }
    _atomic_json(args.results_root / "experiment_summary.json", summary)
    print(
        json.dumps(
            {
                "hyperparameter_winner": summary["hyperparameter_winner"],
                "selected_run": summary["selected_run"],
                "summary": str(args.results_root / "experiment_summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
