from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from lfm25.training_provenance import (
    RUN_MANIFEST_FORMAT,
    TRAINER_CODE_PATHS_BY_ARM,
    adapter_artifact_evidence,
    training_manifest_artifact_binding,
    training_run_manifest_evidence,
)


def _sha(character: str) -> str:
    return character * 64


def _adapter(run_dir: Path, *, weights: bytes = b"synthetic-adapter-weights") -> Path:
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}\n',
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    (adapter / "tokenizer.json").write_text(
        '{"synthetic":"not part of adapter binding"}\n',
        encoding="utf-8",
    )
    return adapter


def _contract(prompt_profile: str) -> dict[str, object]:
    if prompt_profile == "candidate_protocol_v1":
        return {
            "name": "candidate_protocol_v1",
            "profile": "candidate_protocol_v1",
            "version": 1,
            "private_prompt": "DO-NOT-EXPOSE-PROMPT",
        }
    if prompt_profile == "legacy":
        return {
            "contract": "lfm25_short_extraction",
            "profile": "legacy",
            "contract_version": 1,
            "private_prompt": "DO-NOT-EXPOSE-PROMPT",
        }
    return {
        "name": "pocketfinancer",
        "profile": "pocketfinancer",
        "version": 3,
        "private_prompt": "DO-NOT-EXPOSE-PROMPT",
    }


def _trainer_code(prompt_profile: str) -> dict[str, str]:
    arm = "selector" if prompt_profile.startswith("candidate") else "direct"
    return {
        name: _sha(format(index % 16, "x"))
        for index, name in enumerate(TRAINER_CODE_PATHS_BY_ARM[arm], start=1)
    }


def _write_bound_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest["artifact_binding_sha256"] = training_manifest_artifact_binding(manifest)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _manifest(adapter: Path, *, prompt_profile: str = "android") -> dict[str, Any]:
    return {
        "manifest_format": RUN_MANIFEST_FORMAT,
        "base_model": "/private/base/model",
        "base_model_provenance": {
            "files": {
                "config.json": {
                    "path": "/private/base/model/config.json",
                    "bytes": 100,
                    "sha256": _sha("1"),
                },
                "model.safetensors": {
                    "path": "/private/base/model/model.safetensors",
                    "bytes": 200,
                    "sha256": _sha("2"),
                },
                "tokenizer.json": {
                    "path": "/private/base/model/tokenizer.json",
                    "bytes": 300,
                    "sha256": _sha("3"),
                },
            }
        },
        "train_file": "/private/data/train.jsonl",
        "train_sha256": _sha("a"),
        "eval_file": "/private/data/dev.jsonl",
        "eval_sha256": _sha("b"),
        "dataset_report": {
            "filename": "candidate_protocol_v1_report.json",
            "bytes": 456,
            "sha256": _sha("d"),
        },
        "seed": 29,
        "contract": _contract(prompt_profile),
        "model_lock": {
            "filename": "model.lock.json",
            "bytes": 123,
            "sha256": _sha("c"),
        },
        "loss": {
            "mode": "per_example_completion_mean",
            "causal_shift": True,
            "ignore_index": -100,
            "token_reduction": "weighted_mean_per_example",
            "example_reduction": "sample_weighted_mean",
            "weights": {"first_supervised_token_weight": 3.0},
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "in_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "w1",
                "w2",
                "w3",
            ],
        },
        "optimization": {
            "learning_rate": 0.0001,
            "epochs_requested": 6,
            "batch_size": 2,
            "gradient_accumulation": 16,
            "effective_batch_size": 32,
            "max_length": 2304,
            "first_supervised_token_weight": 3.0,
            "warmup_ratio": 0.05,
            "warmup_steps": 2,
            "weight_decay": 0.01,
            "early_stopping_patience": 2,
            "max_grad_norm": 1.0,
            "per_device_eval_batch_size": 2,
            "loss_mode": "per_example_completion_mean",
            "prompt_profile": prompt_profile,
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "full_determinism": True,
        },
        "adapter_artifact": adapter_artifact_evidence(adapter),
        "trainer_code_sha256": _trainer_code(prompt_profile),
        "trainer_state": {
            "best_model_checkpoint": "/private/run/checkpoint-10",
            "best_metric": 0.25,
            "best_eval_log": {
                "metric_name": "eval_loss",
                "metric_value": 0.25,
                "epoch": 2.0,
                "step": 10,
            },
            "final_global_step": 15,
            "final_epoch": 3.0,
            "load_best_model_at_end": True,
            "restored_best_model_checkpoint": "/private/run/checkpoint-10",
            "load_best_model_at_end_restored_best_checkpoint": True,
        },
        "private_samples": [{"sms": "DO-NOT-EXPOSE-SMS"}],
    }


def test_training_manifest_evidence_is_allowlisted_and_private_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    adapter = _adapter(run_dir)
    manifest = _manifest(adapter)
    _write_bound_manifest(run_dir, manifest)

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["present"] is evidence["valid"] is True
    assert evidence["seed"] == 29
    assert evidence["datasets"] == {
        "train_sha256": _sha("a"),
        "eval_sha256": _sha("b"),
        "report": {
            "filename": "candidate_protocol_v1_report.json",
            "bytes": 456,
            "sha256": _sha("d"),
        },
    }
    assert evidence["manifest"]["filename"] == "run_manifest.json"
    assert evidence["artifact_binding_sha256"] == manifest["artifact_binding_sha256"]
    assert evidence["adapter_artifact"] == adapter_artifact_evidence(adapter)
    assert evidence["model_lock"] == {
        "filename": "model.lock.json",
        "bytes": 123,
        "sha256": _sha("c"),
    }
    assert evidence["base_model"]["files"] == {
        "config.json": _sha("1"),
        "model.safetensors": _sha("2"),
        "tokenizer.json": _sha("3"),
    }
    assert evidence["checkpoint_selection"] == {
        "best_step": 10,
        "best_metric_name": "eval_loss",
        "best_metric_value": 0.25,
        "best_epoch": 2.0,
        "final_global_step": 15,
        "final_epoch": 3.0,
        "load_best_model_at_end": True,
        "restored_best_checkpoint": True,
    }
    rendered = json.dumps(evidence)
    assert "/private/" not in rendered
    assert "DO-NOT-EXPOSE" not in rendered
    assert '"sms"' not in rendered


def test_adapter_artifact_and_binding_are_deterministic_and_path_free(
    tmp_path: Path,
) -> None:
    adapter_a = _adapter(tmp_path / "a")
    adapter_b = _adapter(tmp_path / "b")
    (adapter_b / "tokenizer.json").write_text("ignored tokenizer change", encoding="utf-8")

    artifact_a = adapter_artifact_evidence(adapter_a)
    artifact_b = adapter_artifact_evidence(adapter_b)

    assert artifact_a == artifact_b
    assert set(artifact_a["files"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    manifest_a = _manifest(adapter_a)
    manifest_b = deepcopy(manifest_a)
    manifest_b["adapter_artifact"] = artifact_b
    assert training_manifest_artifact_binding(manifest_a) == (
        training_manifest_artifact_binding(manifest_b)
    )
    assert str(tmp_path) not in json.dumps(artifact_a)


def test_candidate_protocol_contract_uses_selector_trainer_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "candidate"
    adapter = _adapter(run_dir)
    _write_bound_manifest(
        run_dir,
        _manifest(adapter, prompt_profile="candidate_protocol_v1"),
    )

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["valid"] is True
    assert evidence["prompt"]["profile"] == "candidate_protocol_v1"
    assert tuple(evidence["trainer_code_sha256"]) == TRAINER_CODE_PATHS_BY_ARM["selector"]


def test_tampered_adapter_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    adapter = _adapter(run_dir)
    _write_bound_manifest(run_dir, _manifest(adapter))
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered-adapter-weights")

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["present"] is True
    assert evidence["valid"] is False
    assert evidence["error"] == "ValueError"


def test_manifest_swapped_onto_another_adapter_is_rejected(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    adapter_a = _adapter(run_a, weights=b"adapter-a")
    adapter_b = _adapter(run_b, weights=b"adapter-b")
    manifest_a = _write_bound_manifest(run_a, _manifest(adapter_a))
    _write_bound_manifest(run_b, _manifest(adapter_b))
    (run_b / "run_manifest.json").write_bytes(manifest_a.read_bytes())

    evidence = training_run_manifest_evidence(adapter_b)

    assert evidence is not None
    assert evidence["present"] is True
    assert evidence["valid"] is False
    assert evidence["error"] == "ValueError"


def test_legacy_current_manifest_contract_is_normalized(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    adapter = _adapter(run_dir)
    _write_bound_manifest(run_dir, _manifest(adapter, prompt_profile="legacy"))

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["valid"] is True
    assert evidence["prompt"] == {
        "profile": "legacy",
        "contract_name": "lfm25_short_extraction",
        "contract_profile": "legacy",
        "contract_version": 1,
        "contract_sha256": evidence["prompt"]["contract_sha256"],
    }


@pytest.mark.parametrize(
    ("case", "private_value"),
    [
        ("restoration_false", False),
        ("checkpoint_mismatch", "/private/customer/checkpoint-11"),
        ("metric_mismatch", 0.26),
    ],
)
def test_checkpoint_restoration_evidence_fails_closed_when_inconsistent(
    tmp_path: Path,
    case: str,
    private_value: object,
) -> None:
    run_dir = tmp_path / case
    adapter = _adapter(run_dir)
    manifest = _manifest(adapter)
    _write_bound_manifest(run_dir, manifest)
    state = manifest["trainer_state"]
    assert isinstance(state, dict)
    if case == "restoration_false":
        state["load_best_model_at_end_restored_best_checkpoint"] = private_value
    elif case == "checkpoint_mismatch":
        state["restored_best_model_checkpoint"] = private_value
    else:
        state["best_metric"] = private_value
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["present"] is True
    assert evidence["valid"] is False
    assert evidence["error"] == "ValueError"
    assert "/private/customer/" not in json.dumps(evidence)


def test_missing_manifest_is_legacy_compatible(tmp_path: Path) -> None:
    adapter = tmp_path / "run" / "adapter"
    adapter.mkdir(parents=True)

    assert training_run_manifest_evidence(adapter) is None


def test_invalid_controlled_string_fails_without_leaking_manifest_values(
    tmp_path: Path,
) -> None:
    secret = "Bank SMS OTP 928104 for account 5544"
    run_dir = tmp_path / "run"
    adapter = _adapter(run_dir)
    manifest = _manifest(adapter)
    _write_bound_manifest(run_dir, manifest)
    optimization = manifest["optimization"]
    assert isinstance(optimization, dict)
    optimization["optimizer"] = secret
    manifest["private_samples"] = [{"sms": secret, "sender": "+15551234567"}]
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["present"] is True
    assert evidence["valid"] is False
    assert evidence["error"] == "ValueError"
    rendered = json.dumps(evidence)
    assert secret not in rendered
    assert "+15551234567" not in rendered
    assert '"sms"' not in rendered


def test_malformed_manifest_fails_private_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        '{"sms":"DO-NOT-EXPOSE-MALFORMED"',
        encoding="utf-8",
    )

    evidence = training_run_manifest_evidence(adapter)

    assert evidence is not None
    assert evidence["present"] is True
    assert evidence["valid"] is False
    assert evidence["error"] == "JSONDecodeError"
    assert "DO-NOT-EXPOSE" not in json.dumps(evidence)


def test_trainer_code_maps_have_exact_controlled_keys() -> None:
    common = (
        "scripts/train_lfm25_lora.py",
        "lfm25/training_provenance.py",
        "lfm25/training_loss.py",
        "lfm25/contract.py",
    )
    assert TRAINER_CODE_PATHS_BY_ARM == {
        "direct": (
            *common,
            "lfm25/prompts.py",
            "lfm25/android_contract.py",
        ),
        "selector": (
            *common,
            "scripts/train_lfm25_candidate_protocol_v1.py",
            "lfm25/candidate_protocol.py",
            "lfm25/candidates.py",
        ),
    }
