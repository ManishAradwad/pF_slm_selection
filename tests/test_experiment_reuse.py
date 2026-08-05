from __future__ import annotations

from pathlib import Path

import pytest

from lfm25.prompts import PRODUCTION_SYSTEM_PROMPT
from lfm25.provenance import (
    code_fingerprints,
    file_sha256,
    fingerprint_file,
    fingerprint_named_files,
)
from lfm25.training_loss import LOSS_MODE
from scripts.run_lfm25_experiments import (
    ADAPTER_FINGERPRINT_FILES,
    REPO_ROOT,
    _common_training_flags,
    _validate_evaluation_reuse,
    _validate_training_reuse,
)


COMMON = {
    "alpha": 32,
    "dropout": 0.05,
    "epochs": 6,
    "batch_size": 8,
    "eval_batch_size": 16,
    "gradient_accumulation": 4,
    "max_length": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
}


def _manifest(base: Path, train: Path, dev: Path, train_hash: str, dev_hash: str):
    import hashlib

    return {
        "base_model": str(base),
        "train_file": str(train),
        "eval_file": str(dev),
        "train_sha256": train_hash,
        "eval_sha256": dev_hash,
        "seed": 17,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
        "optimization": {
            "epochs_requested": 6,
            "batch_size": 8,
            "gradient_accumulation": 4,
            "effective_batch_size": 32,
            "learning_rate": 0.0001,
            "max_length": 512,
            "prompt_profile": "legacy",
            "loss_mode": LOSS_MODE,
            "first_supervised_token_weight": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "early_stopping_patience": 2,
        },
        "contract": {
            "profile": "legacy",
            "contract": "lfm25_short_extraction",
            "contract_version": 1,
            "prompt_sha256": hashlib.sha256(
                PRODUCTION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "loss": {
            "mode": LOSS_MODE,
            "causal_shift": True,
            "ignore_index": -100,
            "token_reduction": "weighted_mean_per_example",
            "example_reduction": "sample_weighted_mean",
            "weights": {
                "sample_weight_field": "sample_weight",
                "default_sample_weight": 1.0,
                "first_supervised_token_weight": 1.0,
                "train_explicit_sample_weight_rows": 0,
                "eval_explicit_sample_weight_rows": 0,
            },
        },
        "eval_stats": {"rows": 2},
    }


def test_training_reuse_validates_config_and_input_hashes(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("train\n", encoding="utf-8")
    dev.write_text("dev\n", encoding="utf-8")
    manifest = _manifest(base, train, dev, file_sha256(train), file_sha256(dev))
    _validate_training_reuse(
        manifest,
        base_model=base.resolve(),
        train=train.resolve(),
        dev=dev.resolve(),
        rank=16,
        learning_rate=0.0001,
        seed=17,
        common=COMMON,
    )
    manifest["seed"] = 29
    with pytest.raises(RuntimeError, match="seed"):
        _validate_training_reuse(
            manifest,
            base_model=base.resolve(),
            train=train.resolve(),
            dev=dev.resolve(),
            rank=16,
            learning_rate=0.0001,
            seed=17,
            common=COMMON,
        )


def test_training_reuse_rejects_changed_dataset(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("train\n", encoding="utf-8")
    dev.write_text("dev\n", encoding="utf-8")
    manifest = _manifest(base, train, dev, "stale", "stale")
    with pytest.raises(RuntimeError, match="train_sha256"):
        _validate_training_reuse(
            manifest,
            base_model=base.resolve(),
            train=train.resolve(),
            dev=dev.resolve(),
            rank=16,
            learning_rate=0.0001,
            seed=17,
            common=COMMON,
        )


def test_training_reuse_rejects_manifest_without_contract_loss_provenance(
    tmp_path: Path,
):
    base = tmp_path / "base"
    base.mkdir()
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("train\n", encoding="utf-8")
    dev.write_text("dev\n", encoding="utf-8")
    manifest = _manifest(base, train, dev, file_sha256(train), file_sha256(dev))
    manifest.pop("contract")
    manifest.pop("loss")
    manifest["optimization"].pop("prompt_profile")
    manifest["optimization"].pop("loss_mode")
    manifest["optimization"].pop("first_supervised_token_weight")

    with pytest.raises(RuntimeError, match="contract.profile"):
        _validate_training_reuse(
            manifest,
            base_model=base.resolve(),
            train=train.resolve(),
            dev=dev.resolve(),
            rank=16,
            learning_rate=0.0001,
            seed=17,
            common=COMMON,
        )


def test_training_flags_pass_explicit_contract_loss_settings():
    common = {
        **COMMON,
        "prompt_profile": "legacy",
        "loss_mode": LOSS_MODE,
        "first_supervised_token_weight": 3.0,
    }
    flags = _common_training_flags(common)
    assert flags[flags.index("--prompt-profile") + 1] == "legacy"
    assert flags[flags.index("--loss-mode") + 1] == LOSS_MODE
    assert flags[flags.index("--first-supervised-token-weight") + 1] == "3.0"

    with pytest.raises(ValueError, match="only supports prompt_profile='legacy'"):
        _common_training_flags({**common, "prompt_profile": "android"})


def _evaluation_fixture(tmp_path: Path):
    dataset = tmp_path / "dev.jsonl"
    dataset.write_text("first\nsecond\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    manifest = {"eval_stats": {"rows": 2}, "eval_sha256": file_sha256(dataset)}
    metrics = {
        "counts": {"rows": 2},
        "provenance": {
            "dataset": fingerprint_file(dataset),
            "adapter": fingerprint_named_files(adapter, ADAPTER_FINGERPRINT_FILES),
            "code_sha256": code_fingerprints(REPO_ROOT),
            "decode": {
                "engine": "transformers",
                "grammar_constrained": False,
                "do_sample": False,
                "repetition_penalty": 1.05,
                "max_new_tokens": 96,
                "seed": 17,
            },
            "row_limit": None,
            "row_count": 2,
        },
    }
    return dataset, adapter, manifest, metrics


def test_evaluation_reuse_validates_dataset_adapter_contract_and_decode(
    tmp_path: Path,
):
    dataset, adapter, manifest, metrics = _evaluation_fixture(tmp_path)
    _validate_evaluation_reuse(
        metrics,
        manifest,
        dataset=dataset,
        adapter=adapter,
        seed=17,
        common=COMMON,
    )

    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="adapter.files"):
        _validate_evaluation_reuse(
            metrics,
            manifest,
            dataset=dataset,
            adapter=adapter,
            seed=17,
            common=COMMON,
        )


def test_evaluation_reuse_rejects_missing_provenance_and_decode_drift(
    tmp_path: Path,
):
    dataset, adapter, manifest, metrics = _evaluation_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="dataset.sha256"):
        _validate_evaluation_reuse(
            {"counts": {"rows": 2}},
            manifest,
            dataset=dataset,
            adapter=adapter,
            seed=17,
            common=COMMON,
        )

    metrics["provenance"]["decode"]["repetition_penalty"] = 1.0
    with pytest.raises(RuntimeError, match="decode.repetition_penalty"):
        _validate_evaluation_reuse(
            metrics,
            manifest,
            dataset=dataset,
            adapter=adapter,
            seed=17,
            common=COMMON,
        )
