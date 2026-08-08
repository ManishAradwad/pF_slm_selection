from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lfm25.candidate_protocol_compare import (
    ComparisonEvidenceError,
    SEEDS,
    compare_metric_files,
    compare_metrics,
    write_report,
)
from scripts.compare_lfm25_candidate_protocol_v1 import main


def _sha(character: str) -> str:
    return character * 64


def _fingerprint(sha256: str) -> dict[str, object]:
    return {"path": "/aggregate/path", "bytes": 10, "sha256": sha256}


def _safe_fingerprint(filename: str, sha256: str) -> dict[str, object]:
    return {"filename": filename, "bytes": 10, "sha256": sha256}


def _files(files: dict[str, str]) -> dict[str, object]:
    return {
        "path": "/aggregate/model",
        "files": {name: {"bytes": 10, "sha256": digest} for name, digest in files.items()},
    }


def _identity(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_sha(relative: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()


MODEL_FILES = {
    "config.json": _sha("1"),
    "model.safetensors": _sha("2"),
    "tokenizer.json": _sha("3"),
}
MODEL_LOCK = _safe_fingerprint("model.lock.json", _sha("0"))
DATA_REPORT = _safe_fingerprint(
    "candidate_protocol_v1_report.json",
    _sha("f"),
)
PROFILES = {
    "candidate": _safe_fingerprint("pocketfinancer-candidate-v1.json", _sha("1")),
    "baseline": _safe_fingerprint("pocketfinancer-android-current.json", _sha("2")),
    "golden_vectors": _safe_fingerprint(
        "candidate_protocol_v1_golden.json",
        _sha("3"),
    ),
}
SHARED_CODE = {
    "lfm25/contract.py": _sha("7"),
    "lfm25/metrics.py": _sha("8"),
}
TRAINER_CODE = {
    "direct": {"scripts/train_lfm25_lora.py": _sha("a")},
    "selector": {"scripts/train_lfm25_candidate_protocol_v1.py": _sha("b")},
}
EVALUATORS = {
    "direct": _sha("5"),
    "selector": _sha("6"),
    "selector_generation_engine": _sha("4"),
    "comparator_module": _repo_sha("lfm25/candidate_protocol_compare.py"),
    "comparator_cli": _repo_sha("scripts/compare_lfm25_candidate_protocol_v1.py"),
}
PLATFORM_GATES = {
    "hf_host_reference_only": True,
    "android_implemented": False,
    "ios_implemented": False,
    "android_runtime_parity": False,
    "ios_runtime_parity": False,
    "gguf_runtime_validated": False,
    "android_device_validated": False,
    "ios_device_validated": False,
    "product_promotion_allowed": False,
}
DIAGNOSTIC_PREFILTER = {
    "enabled": True,
    "n": 203,
    "model_invocations": 115,
    "rejected": 88,
    "rejection_rate": 0.433498,
    "gold_transactions": 114,
    "transactions_passed": 114,
    "transactions_rejected": 0,
    "transaction_recall": 1.0,
    "gold_nulls": 89,
    "nulls_rejected": 88,
    "null_rejection_rate": 0.988764,
    "rejections_by_stage": {
        "currency_amount": 43,
        "masked_account_or_card": 40,
        "otp": 2,
        "transaction_verb": 3,
    },
}


def _candidate_protocol() -> dict[str, object]:
    return {
        "name": "candidate_protocol_v1",
        "version": 1,
        "offset_convention": "utf8_bytes",
        "protocol_module_sha256": _sha("9"),
        "candidate_extractor_sha256": _sha("a"),
        "system_prompt_utf8_sha256": _sha("b"),
        "selector_schema_sha256": _sha("c"),
    }


def _adapter_files(arm: str, seed: int) -> dict[str, dict[str, object]]:
    slot = (0 if arm == "direct" else 8) + SEEDS.index(seed) + 5
    return {
        "adapter_config.json": {"bytes": 10, "sha256": _sha("4")},
        "adapter_model.safetensors": {
            "bytes": 10,
            "sha256": _sha(hex(slot)[2:]),
        },
    }


def _training_run(
    arm: str,
    seed: int,
    prefilter: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    base_files = dict(MODEL_FILES)
    if arm == "direct":
        prompt = {
            "profile": "android",
            "contract_name": "pocketfinancer",
            "contract_profile": "pocketfinancer",
            "contract_version": 3,
            "contract_sha256": _identity(prefilter),
        }
        batch_size, eval_batch_size, accumulation, max_length = 2, 2, 16, 2304
    else:
        prompt = {
            "profile": "candidate_protocol_v1",
            "contract_name": "candidate_protocol_v1",
            "contract_profile": "candidate_protocol_v1",
            "contract_version": 1,
            "contract_sha256": _identity({"profile": "candidate_protocol_v1", **protocol}),
        }
        batch_size, eval_batch_size, accumulation, max_length = 8, 8, 4, 1024
    evidence: dict[str, object] = {
        "seed": seed,
        "datasets": {
            "train_sha256": _sha("d"),
            "report": dict(DATA_REPORT),
            "eval_sha256": _sha("e"),
        },
        "base_model": {
            "files": base_files,
            "identity_sha256": _identity(base_files),
        },
        "prompt": prompt,
        "loss": {
            "mode": "per_example_completion_mean",
            "causal_shift": True,
            "ignore_index": -100,
            "token_reduction": "weighted_mean_per_example",
            "example_reduction": "sample_weighted_mean",
            "first_supervised_token_weight": 3.0,
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
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "effective_batch_size": 32,
            "max_length": max_length,
            "first_supervised_token_weight": 3.0,
            "warmup_ratio": 0.05,
            "warmup_steps": 2,
            "weight_decay": 0.01,
            "early_stopping_patience": 2,
            "max_grad_norm": 1.0,
            "per_device_eval_batch_size": eval_batch_size,
            "loss_mode": "per_example_completion_mean",
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
    }
    evidence["identity_sha256"] = _identity(evidence)
    adapter_files = _adapter_files(arm, seed)
    adapter_payload = {
        "format": "peft_adapter_artifact_v1",
        "files": adapter_files,
    }
    checkpoint = {
        "best_step": seed,
        "best_metric_name": "eval_loss",
        "best_metric_value": 0.5,
        "best_epoch": 1.0,
        "final_global_step": 100,
        "final_epoch": 2.0,
        "load_best_model_at_end": True,
        "restored_best_checkpoint": True,
    }
    evidence.update(
        {
            "model_lock": dict(MODEL_LOCK),
            "adapter_artifact": {
                **adapter_payload,
                "identity_sha256": _identity(adapter_payload),
            },
            "checkpoint_selection": checkpoint,
            "trainer_code_sha256": dict(TRAINER_CODE[arm]),
        }
    )
    adapter_artifact = evidence["adapter_artifact"]
    assert isinstance(adapter_artifact, dict)
    evidence["artifact_binding_sha256"] = _identity(
        {
            "format": "lfm25_training_adapter_binding_v1",
            "training_identity_sha256": evidence["identity_sha256"],
            "model_lock": evidence["model_lock"],
            "adapter_artifact_identity_sha256": adapter_artifact["identity_sha256"],
            "checkpoint_selection": checkpoint,
            "trainer_code_sha256": evidence["trainer_code_sha256"],
        }
    )
    slot = (0 if arm == "direct" else 3) + SEEDS.index(seed)
    return {
        "present": True,
        "valid": True,
        "manifest": {
            "filename": "run_manifest.json",
            "bytes": 10,
            "sha256": _sha("123456"[slot]),
        },
        **evidence,
    }


def _common_provenance(arm: str, seed: int) -> dict[str, object]:
    prefilter = {
        "name": "pocketfinancer",
        "profile": "pocketfinancer",
        "version": 3,
        "contract_sha256": _sha("a"),
    }
    protocol = _candidate_protocol()
    provenance: dict[str, object] = {
        "dataset": {
            **_fingerprint(_sha("d")),
            "row_count": 203,
            "row_limit": None,
        },
        "model": _files(MODEL_FILES),
        "model_lock": dict(MODEL_LOCK),
        "adapter": {
            "path": "/aggregate/adapter",
            "files": _adapter_files(arm, seed),
        },
        "evaluator": _fingerprint(EVALUATORS[arm]),
        "code_sha256": dict(SHARED_CODE),
        "selection_prefilter": {
            "applied": True,
            "part_of_android_current": True,
            "rejected_prediction": "null",
        },
    }
    if arm == "direct":
        provenance.update(
            {
                "decode": {
                    "engine": "transformers_prompt_training_proxy",
                    "grammar_constrained": False,
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 256,
                    "n_ctx": 3072,
                    "seed": seed,
                    "two_pass": False,
                },
                "profile": {
                    "name": "pocketfinancer_hf_training_evaluation",
                    "android_current_prompt_contract": prefilter,
                },
            }
        )
    else:
        provenance.update(
            {
                "pipeline": "pocketfinancer_candidate_protocol_v1_hf",
                "decode": {
                    "engine": "transformers",
                    "do_sample": False,
                    "repeat_penalty": 1.0,
                    "max_new_tokens": 64,
                    "n_ctx": 1024,
                    "seed": seed,
                },
                "prefilter_contract": prefilter,
                "candidate_protocol": protocol,
                "candidate_protocol_code": _fingerprint(_sha("9")),
                "candidate_extractor": _fingerprint(_sha("a")),
                "generation_engine": _fingerprint(EVALUATORS["selector_generation_engine"]),
                "candidate_profile": PROFILES,
                "platform_gates": dict(PLATFORM_GATES),
            }
        )
    provenance["training_run"] = _training_run(arm, seed, prefilter, protocol)
    return provenance


def _metrics(arm: str, seed: int) -> dict[str, object]:
    direct_exact = {17: 80, 29: 82, 43: 81}[seed]
    direct_fp = {17: 4, 29: 3, 43: 5}[seed]
    transaction_exact = direct_exact + (2 if arm == "selector" else 0)
    fp = direct_fp - (1 if arm == "selector" else 0)
    metrics: dict[str, object] = {
        "counts": {
            "rows": 203,
            "transaction_exact": transaction_exact,
            "fp": fp,
        },
        "transaction_only_exact_match": round(transaction_exact / 114, 6),
        "conditional_ghost_rate": round(fp / 89, 6),
        "runtime": {
            "rows": 203,
            "model_invocations": 115,
            "thinking_mode": False if arm == "direct" else "off",
            "batch_size": 8,
            "prefilter_applied": True,
        },
        "provenance": _common_provenance(arm, seed),
        "prefilter": dict(DIAGNOSTIC_PREFILTER),
    }
    if arm == "direct":
        metrics["runtime"]["profile"] = "pocketfinancer"  # type: ignore[index]
    else:
        metrics["runtime"]["model_output_protocol"] = (  # type: ignore[index]
            "candidate_protocol_v1"
        )
        metrics.update(
            {
                "candidate_oracle": {
                    "transactions": 114,
                    "joint_covered": 113,
                    "joint_coverage": 0.991228,
                    "field_coverage": {
                        "amount": 1.0,
                        "account": 1.0,
                        "counterparty": 0.991228,
                    },
                },
                "candidate_protocol_acceptance": {
                    "model_invocations": 115,
                    "accepted_outputs": 115,
                    "rejected_outputs": 0,
                    "strict_schema_acceptance_rate": 1.0,
                    "accepted_transactions": 110,
                    "source_grounded_transactions": 110,
                    "source_grounded_transaction_rate": 1.0,
                },
                "selector_reason_counts": {
                    "accepted_not_transaction": 5,
                    "accepted_transaction": 110,
                },
                "selector_status_counts": {"null": 5, "transaction": 110},
                "conditional_model": {"counts": {"rows": 115, "schema_valid": 115}},
                "hybrid_safety": {"enabled": False, "intervention_counts": {}},
            }
        )
    return metrics


def _matrix() -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    return (
        {seed: _metrics("direct", seed) for seed in SEEDS},
        {seed: _metrics("selector", seed) for seed in SEEDS},
    )


def _trusted_anchors() -> dict[str, object]:
    return {
        "schema_version": 1,
        "diagnostic_dataset": {
            **_safe_fingerprint("extraction_ds.jsonl", _sha("d")),
            "rows": 203,
            "row_limit": None,
        },
        "diagnostic_prefilter": dict(DIAGNOSTIC_PREFILTER),
        "candidate_data": {
            "report": dict(DATA_REPORT),
            "train_sha256": _sha("d"),
            "dev_sha256": _sha("e"),
            "rows": {"train": 152, "dev": 29},
        },
        "model": {
            "lock": dict(MODEL_LOCK),
            "id": "LiquidAI/LFM2.5-350M",
            "revision": "36aa424c15e1bd69acab3380c0854b3d188e1036",
            "files": dict(MODEL_FILES),
        },
        "profiles": PROFILES,
        "prefilter_contract": {
            "name": "pocketfinancer",
            "profile": "pocketfinancer",
            "version": 3,
            "contract_sha256": _sha("a"),
        },
        "candidate_protocol": _candidate_protocol(),
        "shared_code_sha256": dict(SHARED_CODE),
        "evaluator_code_sha256": dict(EVALUATORS),
        "trainer_code_sha256": {arm: dict(values) for arm, values in TRAINER_CODE.items()},
        "platform_gates": dict(PLATFORM_GATES),
    }


def _write_matrix(
    root: Path,
    direct: dict[int, dict[str, object]],
    selector: dict[int, dict[str, object]],
) -> tuple[dict[int, Path], dict[int, Path]]:
    paths: dict[str, dict[int, Path]] = {"direct": {}, "selector": {}}
    for arm, values in (("direct", direct), ("selector", selector)):
        for seed, metrics in values.items():
            directory = root / f"{arm}-s{seed}"
            directory.mkdir(parents=True)
            path = directory / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            paths[arm][seed] = path
    return paths["direct"], paths["selector"]


def test_in_memory_comparison_is_explicitly_non_evidentiary() -> None:
    direct, selector = _matrix()

    report = compare_metrics(direct, selector)

    assert report["report_type"] == "candidate_protocol_v1_non_evidentiary_analysis"
    assert report["evidence_mode"] == "in_memory_non_evidentiary"
    assert report["controlled_hf_gate"] == {
        **report["controlled_hf_gate"],
        "passed": False,
        "evidence_validated": False,
        "criteria_satisfied": True,
    }
    assert "metrics_object_sha256" in report["evidence"]
    assert "metrics_file_sha256" not in report["evidence"]
    assert all(row["delta"]["transaction_exact"] == 2 for row in report["per_seed"])
    assert report["evidence"]["candidate_oracle"] == {
        "amount": {"covered": 114, "total": 114},
        "account": {"covered": 114, "total": 114},
        "counterparty": {"covered": 113, "total": 114},
        "joint": {"covered": 113, "total": 114},
    }
    assert report["product_promotion"]["allowed"] is False
    assert report["unmet_gates"]["fresh_human_gold"]["required_rows"] == 1436
    assert {"selector_gguf", "android_runtime", "ios_runtime"} <= set(report["unmet_gates"])
    assert report["evidence"]["training_data"] == {
        "train_sha256": _sha("d"),
        "dev_sha256": _sha("e"),
    }
    assert set(report["evidence"]["training_manifest_sha256"]) == {
        "direct",
        "selector",
    }


def test_real_per_seed_shortfall_produces_failed_report() -> None:
    direct, selector = _matrix()
    selector[29]["counts"]["transaction_exact"] = direct[29]["counts"][  # type: ignore[index]
        "transaction_exact"
    ]
    selector[29]["transaction_only_exact_match"] = direct[29]["transaction_only_exact_match"]

    report = compare_metrics(direct, selector)

    assert report["controlled_hf_gate"]["passed"] is False
    seed_29 = next(row for row in report["per_seed"] if row["seed"] == 29)
    assert seed_29["checks"]["transaction_exact_strictly_greater"] is False
    assert report["product_promotion"]["allowed"] is False


def test_omitted_zero_counter_fields_are_normalized() -> None:
    direct, selector = _matrix()
    direct_counts = direct[17]["counts"]
    assert isinstance(direct_counts, dict)
    direct_counts["fp"] = 0
    direct_counts["transaction_exact"] = 0
    direct[17]["conditional_ghost_rate"] = 0.0
    direct[17]["transaction_only_exact_match"] = 0.0
    direct_counts.pop("fp")
    direct_counts.pop("transaction_exact")

    report = compare_metrics(direct, selector)

    seed_17 = next(row for row in report["per_seed"] if row["seed"] == 17)
    assert seed_17["direct"]["fp"] == 0
    assert seed_17["direct"]["transaction_exact"] == 0


@pytest.mark.parametrize(
    "key,rate_field",
    [
        ("fp", "conditional_ghost_rate"),
        ("transaction_exact", "transaction_only_exact_match"),
    ],
)
def test_omitted_nonzero_counter_field_fails_closed(key: str, rate_field: str) -> None:
    direct, selector = _matrix()
    direct_counts = direct[17]["counts"]
    assert isinstance(direct_counts, dict)
    direct_counts.pop(key)

    with pytest.raises(ComparisonEvidenceError, match=f"{rate_field} disagrees with counts"):
        compare_metrics(direct, selector)


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "0"])
def test_explicit_invalid_counter_field_fails_closed(value: object) -> None:
    direct, selector = _matrix()
    direct_counts = direct[17]["counts"]
    assert isinstance(direct_counts, dict)
    direct_counts["fp"] = value

    with pytest.raises(ComparisonEvidenceError, match="counts.fp"):
        compare_metrics(direct, selector)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda metrics: metrics["provenance"].pop("candidate_protocol"), "candidate_protocol"),
        (lambda metrics: metrics.pop("candidate_oracle"), "candidate_oracle"),
        (
            lambda metrics: metrics.pop("candidate_protocol_acceptance"),
            "candidate_protocol_acceptance",
        ),
        (
            lambda metrics: metrics["provenance"].pop("training_run"),
            "training_run",
        ),
    ],
)
def test_missing_selector_provenance_or_coverage_fails_closed(mutation, match: str) -> None:
    direct, selector = _matrix()
    mutation(selector[17])

    with pytest.raises(ComparisonEvidenceError, match=match):
        compare_metrics(direct, selector)


def test_dataset_mismatch_is_invalid_controlled_evidence() -> None:
    direct, selector = _matrix()
    selector[43]["provenance"]["dataset"]["sha256"] = _sha("e")  # type: ignore[index]

    with pytest.raises(ComparisonEvidenceError, match="dataset fingerprints differ"):
        compare_metrics(direct, selector)


def test_training_seed_mismatch_fails_closed() -> None:
    direct, selector = _matrix()
    selector[43]["provenance"]["training_run"]["seed"] = 29  # type: ignore[index]

    with pytest.raises(ComparisonEvidenceError, match="training seed"):
        compare_metrics(direct, selector)


def test_training_data_mismatch_is_not_a_controlled_comparison() -> None:
    direct, selector = _matrix()
    training = selector[43]["provenance"]["training_run"]  # type: ignore[index]
    training["datasets"]["train_sha256"] = _sha("f")  # type: ignore[index]
    identity_payload = {
        key: training[key]
        for key in ("seed", "datasets", "base_model", "prompt", "loss", "lora", "optimization")
    }
    training["identity_sha256"] = _identity(identity_payload)  # type: ignore[index]
    training["artifact_binding_sha256"] = _identity(  # type: ignore[index]
        {
            "format": "lfm25_training_adapter_binding_v1",
            "training_identity_sha256": training["identity_sha256"],
            "model_lock": training["model_lock"],
            "adapter_artifact_identity_sha256": training["adapter_artifact"]["identity_sha256"],
            "checkpoint_selection": training["checkpoint_selection"],
            "trainer_code_sha256": training["trainer_code_sha256"],
        }
    )

    with pytest.raises(ComparisonEvidenceError, match="training train/dev fingerprints differ"):
        compare_metrics(direct, selector)


def test_training_hyperparameter_mismatch_fails_closed() -> None:
    direct, selector = _matrix()
    selector[17]["provenance"]["training_run"]["optimization"][  # type: ignore[index]
        "learning_rate"
    ] = 0.0002

    with pytest.raises(ComparisonEvidenceError, match="learning_rate"):
        compare_metrics(direct, selector)


def test_reused_adapter_identity_is_not_a_controlled_matrix() -> None:
    direct, selector = _matrix()
    selector[43]["provenance"]["adapter"] = selector[29]["provenance"][  # type: ignore[index]
        "adapter"
    ]

    with pytest.raises(ComparisonEvidenceError, match="not bound to the evaluated adapter"):
        compare_metrics(direct, selector)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("prefilter", False, "prefilter is not the canonical aggregate"),
        ("batch_size", 4, "evaluation batch size"),
        ("max_new_tokens", 63, "decode context or completion budget"),
    ],
)
def test_evaluation_lock_mismatch_fails_closed(field: str, value: object, match: str) -> None:
    direct, selector = _matrix()
    if field == "prefilter":
        selector[17]["prefilter"]["enabled"] = value  # type: ignore[index]
    elif field == "batch_size":
        selector[17]["runtime"][field] = value  # type: ignore[index]
    else:
        selector[17]["provenance"]["decode"][field] = value  # type: ignore[index]

    with pytest.raises(ComparisonEvidenceError, match=match):
        compare_metrics(direct, selector)


def test_file_comparison_never_reads_samples_or_projects_private_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct, selector = _matrix()
    selector[17]["accidental_private_payload"] = {"sms": "DO-NOT-READ-OR-REPORT"}
    direct_paths, selector_paths = _write_matrix(tmp_path, direct, selector)
    for path in (*direct_paths.values(), *selector_paths.values()):
        (path.parent / "samples.jsonl").write_text(
            '{"sms":"ADJACENT-PRIVATE-SAMPLE"}\n', encoding="utf-8"
        )

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    metric_reads: dict[Path, int] = {}

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name == "samples.jsonl":
            raise AssertionError("comparison attempted to read samples")
        return original_read_text(path, *args, **kwargs)

    def counted_read_bytes(path: Path, *args, **kwargs):
        if path.name == "metrics.json":
            metric_reads[path] = metric_reads.get(path, 0) + 1
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    report = compare_metric_files(
        direct_paths,
        selector_paths,
        trusted_anchors=_trusted_anchors(),
    )
    rendered = json.dumps(report)

    assert report["controlled_hf_gate"]["passed"] is True
    assert "DO-NOT-READ-OR-REPORT" not in rendered
    assert "ADJACENT-PRIVATE-SAMPLE" not in rendered
    assert '"sms"' not in rendered

    assert set(metric_reads) == {*direct_paths.values(), *selector_paths.values()}
    assert set(metric_reads.values()) == {1}


@pytest.mark.parametrize(
    "path",
    [
        ("diagnostic_dataset", "sha256"),
        ("diagnostic_prefilter", "model_invocations"),
        ("candidate_data", "train_sha256"),
        ("candidate_data", "dev_sha256"),
        ("candidate_data", "report", "sha256"),
        ("model", "revision"),
        ("model", "lock", "sha256"),
        ("model", "files", "config.json"),
        ("profiles", "candidate", "sha256"),
        ("profiles", "baseline", "sha256"),
        ("profiles", "golden_vectors", "sha256"),
        ("prefilter_contract", "contract_sha256"),
        ("candidate_protocol", "system_prompt_utf8_sha256"),
        ("shared_code_sha256", "lfm25/contract.py"),
        ("evaluator_code_sha256", "direct"),
        ("evaluator_code_sha256", "selector"),
        ("evaluator_code_sha256", "selector_generation_engine"),
        ("evaluator_code_sha256", "comparator_module"),
        ("evaluator_code_sha256", "comparator_cli"),
        ("trainer_code_sha256", "direct", "scripts/train_lfm25_lora.py"),
        (
            "trainer_code_sha256",
            "selector",
            "scripts/train_lfm25_candidate_protocol_v1.py",
        ),
        ("platform_gates", "ios_runtime_parity"),
    ],
)
def test_each_trusted_anchor_mismatch_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
) -> None:
    direct, selector = _matrix()
    direct_paths, selector_paths = _write_matrix(tmp_path, direct, selector)
    anchors = copy.deepcopy(_trusted_anchors())
    target: dict[str, object] = anchors
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    old = target[path[-1]]
    if isinstance(old, bool):
        target[path[-1]] = not old
    elif path[-1] == "revision":
        target[path[-1]] = "untrusted-revision"
    else:
        target[path[-1]] = _sha("0") if old != _sha("0") else _sha("1")

    with pytest.raises(ComparisonEvidenceError):
        compare_metric_files(
            direct_paths,
            selector_paths,
            trusted_anchors=anchors,
        )


def test_atomic_writer_and_cli_refuse_overwrite_without_force(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct, selector = _matrix()
    direct_paths, selector_paths = _write_matrix(tmp_path, direct, selector)
    report = compare_metric_files(
        direct_paths,
        selector_paths,
        trusted_anchors=_trusted_anchors(),
    )
    output = tmp_path / "comparison.json"
    write_report(report, output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(report, output)

    arguments = [
        *[item for seed in SEEDS for item in (f"--direct-s{seed}", str(direct_paths[seed]))],
        *[item for seed in SEEDS for item in (f"--selector-s{seed}", str(selector_paths[seed]))],
        "--output",
        str(output),
        "--force",
    ]
    monkeypatch.setattr(
        "scripts.compare_lfm25_candidate_protocol_v1.candidate_comparison_anchors",
        lambda *_args: _trusted_anchors(),
    )
    assert main(arguments) == 0
    stdout = capsys.readouterr().out
    assert "candidate_protocol_v1_controlled_hf_comparison" in stdout
    assert "sms" not in stdout.casefold()


def test_templates_expand_predictably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct, selector = _matrix()
    direct_paths, selector_paths = _write_matrix(tmp_path, direct, selector)
    output = tmp_path / "template-comparison.json"

    monkeypatch.setattr(
        "scripts.compare_lfm25_candidate_protocol_v1.candidate_comparison_anchors",
        lambda *_args: _trusted_anchors(),
    )
    result = main(
        [
            "--direct-template",
            str(tmp_path / "direct-s{seed}" / "metrics.json"),
            "--selector-template",
            str(tmp_path / "selector-s{seed}" / "metrics.json"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.is_file()
    assert set(direct_paths) == set(selector_paths) == set(SEEDS)
