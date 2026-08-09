"""Aggregate-only controlled comparison for Candidate Protocol V1.

The comparison intentionally reads only the six requested ``metrics.json``
files.  It never opens evaluator sample files, and it projects only aggregate
counts and cryptographic identities into its report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SEEDS = (17, 29, 43)
DIAGNOSTIC_ROWS = 203
PROTOCOL_NAME = "candidate_protocol_v1"
MODEL_REVISION = "36aa424c15e1bd69acab3380c0854b3d188e1036"
LORA_TARGETS = (
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
)
EVALUATION_LOCKS = {
    "direct": {
        "engine": "transformers_prompt_training_proxy",
        "batch_size": 8,
        "max_new_tokens": 256,
        "n_ctx": 3072,
    },
    "selector": {
        "engine": "transformers",
        "batch_size": 8,
        "max_new_tokens": 64,
        "n_ctx": 1024,
    },
}
TRAINING_ARM_LOCKS = {
    "direct": {
        "prompt_profile": "android",
        "contract": ("pocketfinancer", "pocketfinancer", 3),
        "batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation": 16,
        "max_length": 2304,
    },
    "selector": {
        "prompt_profile": PROTOCOL_NAME,
        "contract": (PROTOCOL_NAME, PROTOCOL_NAME, 1),
        "batch_size": 8,
        "eval_batch_size": 8,
        "gradient_accumulation": 4,
        "max_length": 1024,
    },
}

ORACLE_FLOORS = {
    "amount": (114, 114),
    "account": (114, 114),
    "counterparty": (113, 114),
    "joint": (113, 114),
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
_ADAPTER_WEIGHT_RE = re.compile(
    r"adapter_model(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)"
    r"(?:\.index\.json)?"
)


class ComparisonEvidenceError(RuntimeError):
    """Raised when aggregate evidence is missing, malformed, or uncontrolled."""


@dataclass(frozen=True)
class _RunEvidence:
    arm: str
    seed: int
    metrics_sha256: str
    rows: int
    model_invocations: int
    transaction_exact: int
    fp: int
    evaluation_batch_size: int
    prefilter: dict[str, Any]
    training: dict[str, Any]
    dataset_sha256: str
    dataset_bytes: int
    model_files: dict[str, str]
    model_lock: dict[str, Any]
    adapter_identity: str
    evaluator_sha256: str
    generation_engine_sha256: str | None
    code_sha256: dict[str, str]
    decode: dict[str, Any]
    prefilter_contract: dict[str, Any]
    oracle: dict[str, dict[str, int]] | None
    acceptance: dict[str, int | float] | None
    protocol_provenance: dict[str, Any] | None

    candidate_profile: dict[str, dict[str, Any]] | None
    platform_gates: dict[str, bool] | None


def _fail(label: str, field: str) -> ComparisonEvidenceError:
    return ComparisonEvidenceError(f"{label}: invalid or missing {field}")


def _mapping(value: Any, label: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(label, field)
    return value


def _integer(value: Any, label: str, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(label, field)
    return value


def _counter_integer(counts: Mapping[str, Any], key: str, label: str) -> int:
    """Read a Counter-derived metric, whose zero-valued keys are omitted."""
    return _integer(counts.get(key, 0), label, f"counts.{key}")


def _number(value: Any, label: str, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise _fail(label, field)
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise _fail(label, field) from exc
    if not result.is_finite():
        raise _fail(label, field)
    return result


def _sha256(value: Any, label: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(label, field)
    return value


def _json_identity(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_metrics(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if path.name != "metrics.json":
        raise ComparisonEvidenceError(f"{label}: input must be named metrics.json")
    if not path.is_file():
        raise ComparisonEvidenceError(f"{label}: metrics.json does not exist")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonEvidenceError(f"{label}: metrics.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ComparisonEvidenceError(f"{label}: metrics.json must contain an object")
    metrics_sha256 = hashlib.sha256(payload).hexdigest()
    return value, metrics_sha256


def _fingerprinted_files(value: Any, label: str, field: str) -> dict[str, str]:
    fingerprint = _mapping(value, label, field)
    files = _mapping(fingerprint.get("files"), label, f"{field}.files")
    result: dict[str, str] = {}
    for name, evidence in files.items():
        if not isinstance(name, str) or not name:
            raise _fail(label, f"{field}.files")
        item = _mapping(evidence, label, f"{field}.files")
        result[name] = _sha256(
            item.get("sha256"),
            label,
            f"{field}.files.{name}.sha256",
        )
    if not result:
        raise _fail(label, f"{field}.files")
    return result


def _fingerprint_sha(value: Any, label: str, field: str) -> str:
    fingerprint = _mapping(value, label, field)
    return _sha256(fingerprint.get("sha256"), label, f"{field}.sha256")


def _safe_file_fingerprint(
    value: Any,
    label: str,
    field: str,
    *,
    filename: str,
) -> dict[str, Any]:
    fingerprint = _mapping(value, label, field)
    if fingerprint.get("filename") != filename:
        raise _fail(label, f"{field}.filename")
    return {
        "filename": filename,
        "bytes": _integer(
            fingerprint.get("bytes"),
            label,
            f"{field}.bytes",
            minimum=1,
        ),
        "sha256": _sha256(
            fingerprint.get("sha256"),
            label,
            f"{field}.sha256",
        ),
    }


def _sha_mapping(value: Any, label: str, field: str) -> dict[str, str]:
    values = _mapping(value, label, field)
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise _fail(label, field)
    return {
        name: _sha256(digest, label, f"{field}.{name}") for name, digest in sorted(values.items())
    }


def _adapter_identity(value: Any, label: str) -> str:
    fingerprint = _mapping(value, label, "provenance.adapter")
    file_values = _mapping(
        fingerprint.get("files"),
        label,
        "provenance.adapter.files",
    )
    files: dict[str, dict[str, Any]] = {}
    for raw_name, raw_evidence in sorted(file_values.items()):
        if not isinstance(raw_name, str):
            raise _fail(label, "provenance.adapter.files")
        if raw_name != "adapter_config.json" and _ADAPTER_WEIGHT_RE.fullmatch(raw_name) is None:
            continue
        evidence = _mapping(
            raw_evidence,
            label,
            f"provenance.adapter.files.{raw_name}",
        )
        files[raw_name] = {
            "bytes": _integer(
                evidence.get("bytes"),
                label,
                f"provenance.adapter.files.{raw_name}.bytes",
                minimum=1,
            ),
            "sha256": _sha256(
                evidence.get("sha256"),
                label,
                f"provenance.adapter.files.{raw_name}.sha256",
            ),
        }
    weight_names = {
        name
        for name in files
        if _ADAPTER_WEIGHT_RE.fullmatch(name) is not None and not name.endswith(".index.json")
    }
    if "adapter_config.json" not in files or not weight_names:
        raise _fail(label, "provenance.adapter required files")
    payload = {"format": "peft_adapter_artifact_v1", "files": files}
    return _json_identity(payload)


def _code_identity(value: Any, label: str) -> dict[str, str]:
    code = _mapping(value, label, "provenance.code_sha256")
    if not code:
        raise _fail(label, "provenance.code_sha256")
    if any(not isinstance(name, str) or not name for name in code):
        raise _fail(label, "provenance.code_sha256")
    return {
        name: _sha256(digest, label, f"provenance.code_sha256.{name}")
        for name, digest in sorted(code.items())
    }


def _decode_evidence(
    value: Any,
    runtime: Mapping[str, Any],
    arm: str,
    seed: int,
    label: str,
) -> dict[str, Any]:
    decode = _mapping(value, label, "provenance.decode")
    if decode.get("do_sample") is not False:
        raise _fail(label, "provenance.decode.do_sample=false")
    if _integer(decode.get("seed"), label, "provenance.decode.seed") != seed:
        raise ComparisonEvidenceError(f"{label}: provenance.decode.seed does not match its slot")
    engine = decode.get("engine")
    lock = EVALUATION_LOCKS[arm]
    if engine != lock["engine"]:
        raise ComparisonEvidenceError(
            f"{label}: decode engine does not match the locked {arm} evaluator"
        )
    penalty_key = "repetition_penalty" if "repetition_penalty" in decode else "repeat_penalty"
    penalty = _number(decode.get(penalty_key), label, f"provenance.decode.{penalty_key}")
    if penalty != Decimal("1.0"):
        raise ComparisonEvidenceError(f"{label}: decode repetition penalty is not the locked value")
    max_new_tokens = _integer(
        decode.get("max_new_tokens"), label, "provenance.decode.max_new_tokens", minimum=1
    )
    n_ctx = _integer(decode.get("n_ctx"), label, "provenance.decode.n_ctx", minimum=1)
    if n_ctx <= max_new_tokens:
        raise ComparisonEvidenceError(f"{label}: decode context does not exceed completion budget")
    if max_new_tokens != lock["max_new_tokens"] or n_ctx != lock["n_ctx"]:
        raise ComparisonEvidenceError(
            f"{label}: decode context or completion budget differs from the locked {arm} arm"
        )
    if decode.get("two_pass", False) is not False:
        raise ComparisonEvidenceError(f"{label}: two-pass/thinking decode is enabled")
    thinking_mode = runtime.get("thinking_mode")
    if thinking_mode not in (False, "off"):
        raise ComparisonEvidenceError(f"{label}: runtime thinking mode is not off")
    if arm == "direct" and decode.get("grammar_constrained") is not False:
        raise _fail(label, "provenance.decode.grammar_constrained=false")
    return {
        "engine": engine,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
        "n_ctx": n_ctx,
        "two_pass": False,
        "thinking_mode": "off",
    }


def _ratio_count(value: Any, total: int, label: str, field: str) -> int:
    ratio = _number(value, label, field)
    if ratio < 0 or ratio > 1:
        raise _fail(label, field)
    raw = ratio * total
    rounded = int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    # Evaluators currently round ratios to six decimal places.  This tolerance
    # accepts only a count compatible with such rounding, not an arbitrary rate.
    if abs(raw - rounded) > Decimal("0.0001"):
        raise ComparisonEvidenceError(f"{label}: {field} cannot be resolved to a count")
    return rounded


def _score_rate_count(value: Any, total: int, label: str, field: str) -> int:
    if total == 0:
        if value is not None:
            raise _fail(label, field)
        return 0
    return _ratio_count(value, total, label, field)


def _coverage_entry(
    oracle: Mapping[str, Any],
    field: str,
    total: int,
    label: str,
) -> dict[str, int]:
    candidates: list[int] = []
    field_counts = oracle.get("field_counts")
    if isinstance(field_counts, Mapping) and field in field_counts:
        entry = field_counts[field]
        if isinstance(entry, Mapping):
            entry_total = _integer(entry.get("total"), label, f"candidate_oracle.{field}.total")
            if entry_total != total:
                raise ComparisonEvidenceError(f"{label}: conflicting oracle totals")
            candidates.append(
                _integer(entry.get("covered"), label, f"candidate_oracle.{field}.covered")
            )
        else:
            candidates.append(_integer(entry, label, f"candidate_oracle.{field}_covered"))
    explicit = oracle.get(f"{field}_covered")
    if explicit is not None:
        candidates.append(_integer(explicit, label, f"candidate_oracle.{field}_covered"))
    field_coverage = oracle.get("field_coverage")
    if isinstance(field_coverage, Mapping) and field in field_coverage:
        entry = field_coverage[field]
        if isinstance(entry, Mapping):
            entry_total = _integer(entry.get("total"), label, f"candidate_oracle.{field}.total")
            if entry_total != total:
                raise ComparisonEvidenceError(f"{label}: conflicting oracle totals")
            candidates.append(
                _integer(entry.get("covered"), label, f"candidate_oracle.{field}.covered")
            )
        else:
            candidates.append(
                _ratio_count(entry, total, label, f"candidate_oracle.field_coverage.{field}")
            )
    if not candidates:
        raise _fail(label, f"candidate_oracle.{field} coverage")
    if len(set(candidates)) != 1 or candidates[0] > total:
        raise ComparisonEvidenceError(f"{label}: inconsistent candidate_oracle.{field} coverage")
    return {"covered": candidates[0], "total": total}


def _oracle_coverage(value: Any, label: str) -> dict[str, dict[str, int]]:
    oracle = _mapping(value, label, "candidate_oracle")
    total_value = oracle.get("transactions", oracle.get("total"))
    total = _integer(total_value, label, "candidate_oracle.transactions", minimum=1)
    result = {
        field: _coverage_entry(oracle, field, total, label)
        for field in ("amount", "account", "counterparty")
    }
    joint_candidates: list[int] = []
    if "joint_covered" in oracle:
        joint_candidates.append(
            _integer(oracle.get("joint_covered"), label, "candidate_oracle.joint_covered")
        )
    if "joint_coverage" in oracle:
        joint_candidates.append(
            _ratio_count(
                oracle.get("joint_coverage"),
                total,
                label,
                "candidate_oracle.joint_coverage",
            )
        )
    if not joint_candidates:
        raise _fail(label, "candidate_oracle.joint coverage")
    if len(set(joint_candidates)) != 1 or joint_candidates[0] > total:
        raise ComparisonEvidenceError(f"{label}: inconsistent candidate_oracle.joint coverage")
    result["joint"] = {"covered": joint_candidates[0], "total": total}
    return result


def _protocol_provenance(value: Any, provenance: Mapping[str, Any], label: str) -> dict[str, Any]:
    protocol = dict(_mapping(value, label, "provenance.candidate_protocol"))
    if protocol.get("name") != PROTOCOL_NAME or protocol.get("version") != 1:
        raise ComparisonEvidenceError(f"{label}: Candidate Protocol identity is not V1")
    if protocol.get("offset_convention") != "utf8_bytes":
        raise _fail(label, "provenance.candidate_protocol.offset_convention")
    for key in (
        "protocol_module_sha256",
        "candidate_extractor_sha256",
        "system_prompt_utf8_sha256",
        "selector_schema_sha256",
    ):
        _sha256(protocol.get(key), label, f"provenance.candidate_protocol.{key}")
    protocol_code = _fingerprint_sha(
        provenance.get("candidate_protocol_code"),
        label,
        "provenance.candidate_protocol_code",
    )
    extractor_code = _fingerprint_sha(
        provenance.get("candidate_extractor"),
        label,
        "provenance.candidate_extractor",
    )
    if protocol["protocol_module_sha256"] != protocol_code:
        raise ComparisonEvidenceError(f"{label}: Candidate Protocol module hashes disagree")
    if protocol["candidate_extractor_sha256"] != extractor_code:
        raise ComparisonEvidenceError(f"{label}: candidate extractor hashes disagree")
    return protocol


def _prefilter_summary(value: Any, label: str, field: str) -> dict[str, Any]:
    summary = _mapping(value, label, field)
    expected_keys = {
        "enabled",
        "n",
        "model_invocations",
        "rejected",
        "rejection_rate",
        "gold_transactions",
        "transactions_passed",
        "transactions_rejected",
        "transaction_recall",
        "gold_nulls",
        "nulls_rejected",
        "null_rejection_rate",
        "rejections_by_stage",
    }
    if set(summary) != expected_keys or summary.get("enabled") is not True:
        raise ComparisonEvidenceError(f"{label}: {field} is not the canonical aggregate")
    integers = {
        key: _integer(summary.get(key), label, f"{field}.{key}")
        for key in (
            "n",
            "model_invocations",
            "rejected",
            "gold_transactions",
            "transactions_passed",
            "transactions_rejected",
            "gold_nulls",
            "nulls_rejected",
        )
    }
    rows = integers["n"]
    transactions = integers["gold_transactions"]
    nulls = integers["gold_nulls"]
    invocations = integers["model_invocations"]
    rejected = integers["rejected"]
    if (
        transactions + nulls != rows
        or invocations + rejected != rows
        or integers["transactions_passed"] + integers["transactions_rejected"] != transactions
        or integers["nulls_rejected"] > nulls
        or integers["transactions_rejected"] + integers["nulls_rejected"] != rejected
    ):
        raise ComparisonEvidenceError(f"{label}: {field} aggregate counts are inconsistent")

    def rate(key: str, numerator: int, denominator: int) -> float | None:
        observed = summary.get(key)
        if denominator == 0:
            if observed is not None:
                raise _fail(label, f"{field}.{key}")
            return None
        expected = Decimal(str(round(numerator / denominator, 6)))
        if _number(observed, label, f"{field}.{key}") != expected:
            raise ComparisonEvidenceError(f"{label}: {field}.{key} disagrees with counts")
        return float(expected)

    stages_value = _mapping(
        summary.get("rejections_by_stage"),
        label,
        f"{field}.rejections_by_stage",
    )
    allowed_stages = {
        "personal_mobile_sender",
        "currency_amount",
        "masked_account_or_card",
        "transaction_verb",
        "otp",
        "collect_or_mandate_request",
    }
    if set(stages_value) - allowed_stages:
        raise _fail(label, f"{field}.rejections_by_stage")
    stages = {
        stage: _integer(count, label, f"{field}.rejections_by_stage.{stage}")
        for stage, count in sorted(stages_value.items())
    }
    if sum(stages.values()) != rejected:
        raise ComparisonEvidenceError(f"{label}: {field} rejection stages disagree")
    return {
        "enabled": True,
        **integers,
        "rejection_rate": rate("rejection_rate", rejected, rows),
        "transaction_recall": rate(
            "transaction_recall",
            integers["transactions_passed"],
            transactions,
        ),
        "null_rejection_rate": rate(
            "null_rejection_rate",
            integers["nulls_rejected"],
            nulls,
        ),
        "rejections_by_stage": stages,
    }


def _prefilter_evidence(
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
    rows: int,
    invocations: int,
    label: str,
) -> dict[str, Any]:
    if runtime.get("prefilter_applied") is not True:
        raise ComparisonEvidenceError(f"{label}: PocketFinancer prefilter was not applied")
    summary = _prefilter_summary(metrics.get("prefilter"), label, "prefilter")
    selection = _mapping(
        provenance.get("selection_prefilter"),
        label,
        "provenance.selection_prefilter",
    )
    if (
        selection.get("applied") is not True
        or selection.get("part_of_android_current") is not True
        or selection.get("rejected_prediction") != "null"
    ):
        raise ComparisonEvidenceError(f"{label}: prefilter provenance is not app-aligned")
    if summary["n"] != rows or summary["model_invocations"] != invocations:
        raise ComparisonEvidenceError(f"{label}: prefilter aggregates disagree with runtime counts")
    return summary


def _expect_number(
    value: Mapping[str, Any],
    key: str,
    expected: int | float,
    label: str,
    field: str,
) -> None:
    if _number(value.get(key), label, f"{field}.{key}") != Decimal(str(expected)):
        raise ComparisonEvidenceError(f"{label}: {field}.{key} differs from the locked value")


def _training_adapter_artifact(value: Any, label: str) -> str:
    field = "provenance.training_run.adapter_artifact"
    artifact = _mapping(value, label, field)
    if artifact.get("format") != "peft_adapter_artifact_v1":
        raise _fail(label, f"{field}.format")
    values = _mapping(artifact.get("files"), label, f"{field}.files")
    files: dict[str, dict[str, Any]] = {}
    for raw_name, raw_evidence in sorted(values.items()):
        if not isinstance(raw_name, str) or (
            raw_name != "adapter_config.json" and _ADAPTER_WEIGHT_RE.fullmatch(raw_name) is None
        ):
            raise _fail(label, f"{field}.files")
        evidence = _mapping(raw_evidence, label, f"{field}.files.{raw_name}")
        files[raw_name] = {
            "bytes": _integer(
                evidence.get("bytes"),
                label,
                f"{field}.files.{raw_name}.bytes",
                minimum=1,
            ),
            "sha256": _sha256(
                evidence.get("sha256"),
                label,
                f"{field}.files.{raw_name}.sha256",
            ),
        }
    weights = {
        name
        for name in files
        if _ADAPTER_WEIGHT_RE.fullmatch(name) is not None and not name.endswith(".index.json")
    }
    if "adapter_config.json" not in files or not weights:
        raise _fail(label, f"{field}.files")
    payload = {"format": "peft_adapter_artifact_v1", "files": files}
    identity = _sha256(
        artifact.get("identity_sha256"),
        label,
        f"{field}.identity_sha256",
    )
    if identity != _json_identity(payload):
        raise ComparisonEvidenceError(
            f"{label}: training adapter artifact identity is inconsistent"
        )
    return identity


def _checkpoint_selection(value: Any, label: str) -> dict[str, Any]:
    field = "provenance.training_run.checkpoint_selection"
    checkpoint = _mapping(value, label, field)
    expected_keys = {
        "best_step",
        "best_metric_name",
        "best_metric_value",
        "best_epoch",
        "final_global_step",
        "final_epoch",
        "load_best_model_at_end",
        "restored_best_checkpoint",
    }
    if set(checkpoint) != expected_keys:
        raise _fail(label, field)
    if checkpoint.get("best_metric_name") != "eval_loss":
        raise _fail(label, f"{field}.best_metric_name")
    if (
        checkpoint.get("load_best_model_at_end") is not True
        or checkpoint.get("restored_best_checkpoint") is not True
    ):
        raise ComparisonEvidenceError(
            f"{label}: training did not restore the selected best checkpoint"
        )
    best_step = _integer(
        checkpoint.get("best_step"),
        label,
        f"{field}.best_step",
        minimum=1,
    )
    final_step = _integer(
        checkpoint.get("final_global_step"),
        label,
        f"{field}.final_global_step",
        minimum=1,
    )
    if best_step > final_step:
        raise ComparisonEvidenceError(f"{label}: best checkpoint step is impossible")
    for key in ("best_metric_value", "best_epoch", "final_epoch"):
        _number(checkpoint.get(key), label, f"{field}.{key}")
    return {
        "best_step": best_step,
        "best_metric_name": "eval_loss",
        "best_metric_value": checkpoint["best_metric_value"],
        "best_epoch": checkpoint["best_epoch"],
        "final_global_step": final_step,
        "final_epoch": checkpoint["final_epoch"],
        "load_best_model_at_end": True,
        "restored_best_checkpoint": True,
    }


def _training_run_evidence(
    value: Any,
    *,
    arm: str,
    seed: int,
    model_files: Mapping[str, str],
    expected_contract_sha256: str,
    label: str,
) -> dict[str, Any]:
    training = _mapping(value, label, "provenance.training_run")
    if training.get("present") is not True or training.get("valid") is not True:
        raise ComparisonEvidenceError(f"{label}: valid run_manifest.json evidence is required")
    manifest = _mapping(training.get("manifest"), label, "provenance.training_run.manifest")
    if manifest.get("filename") != "run_manifest.json":
        raise _fail(label, "provenance.training_run.manifest.filename")
    manifest_sha256 = _sha256(
        manifest.get("sha256"),
        label,
        "provenance.training_run.manifest.sha256",
    )
    _integer(
        manifest.get("bytes"),
        label,
        "provenance.training_run.manifest.bytes",
        minimum=1,
    )
    if _integer(training.get("seed"), label, "provenance.training_run.seed") != seed:
        raise ComparisonEvidenceError(f"{label}: training seed does not match its matrix slot")

    datasets = _mapping(training.get("datasets"), label, "provenance.training_run.datasets")
    train_sha256 = _sha256(
        datasets.get("train_sha256"), label, "provenance.training_run.datasets.train_sha256"
    )
    eval_sha256 = _sha256(
        datasets.get("eval_sha256"), label, "provenance.training_run.datasets.eval_sha256"
    )
    data_report = _safe_file_fingerprint(
        datasets.get("report"),
        label,
        "provenance.training_run.datasets.report",
        filename="candidate_protocol_v1_report.json",
    )
    if set(datasets) != {"train_sha256", "eval_sha256", "report"}:
        raise _fail(label, "provenance.training_run.datasets")

    base_model = _mapping(training.get("base_model"), label, "provenance.training_run.base_model")
    base_files_value = _mapping(
        base_model.get("files"), label, "provenance.training_run.base_model.files"
    )
    base_files = {
        str(name): _sha256(
            digest,
            label,
            f"provenance.training_run.base_model.files.{name}",
        )
        for name, digest in base_files_value.items()
        if isinstance(name, str) and name
    }
    if len(base_files) != len(base_files_value) or not base_files:
        raise _fail(label, "provenance.training_run.base_model.files")
    base_identity = _sha256(
        base_model.get("identity_sha256"),
        label,
        "provenance.training_run.base_model.identity_sha256",
    )
    if base_identity != _json_identity(base_files):
        raise ComparisonEvidenceError(f"{label}: training base-model identity is inconsistent")
    common_names = set(base_files) & set(model_files)
    weight_names = {name for name in common_names if name.endswith((".safetensors", ".bin"))}
    tokenizer_names = common_names & {"tokenizer.json", "tokenizer.model"}
    if "config.json" not in common_names or not weight_names or not tokenizer_names:
        raise ComparisonEvidenceError(
            f"{label}: training and evaluation model evidence is incomplete"
        )
    if any(base_files[name] != model_files[name] for name in common_names):
        raise ComparisonEvidenceError(f"{label}: training and evaluation base models differ")

    arm_lock = TRAINING_ARM_LOCKS[arm]
    prompt = _mapping(training.get("prompt"), label, "provenance.training_run.prompt")
    contract_identity = (
        prompt.get("contract_name"),
        prompt.get("contract_profile"),
        prompt.get("contract_version"),
    )
    if prompt.get("profile") != arm_lock["prompt_profile"]:
        raise ComparisonEvidenceError(f"{label}: training prompt profile is not locked")
    if contract_identity != arm_lock["contract"]:
        raise ComparisonEvidenceError(f"{label}: training prompt contract is not locked")
    contract_sha256 = _sha256(
        prompt.get("contract_sha256"),
        label,
        "provenance.training_run.prompt.contract_sha256",
    )
    if contract_sha256 != expected_contract_sha256:
        raise ComparisonEvidenceError(f"{label}: training and evaluation prompt contracts differ")

    loss = _mapping(training.get("loss"), label, "provenance.training_run.loss")
    expected_loss = {
        "mode": "per_example_completion_mean",
        "causal_shift": True,
        "ignore_index": -100,
        "token_reduction": "weighted_mean_per_example",
        "example_reduction": "sample_weighted_mean",
    }
    if any(loss.get(key) != expected for key, expected in expected_loss.items()):
        raise ComparisonEvidenceError(f"{label}: completion-only loss identity is not locked")
    _expect_number(loss, "first_supervised_token_weight", 3.0, label, "training.loss")

    lora = _mapping(training.get("lora"), label, "provenance.training_run.lora")
    if lora.get("rank") != 16 or lora.get("alpha") != 32:
        raise ComparisonEvidenceError(f"{label}: LoRA rank/alpha differs from the lock")
    _expect_number(lora, "dropout", 0.05, label, "training.lora")
    if tuple(lora.get("target_modules", ())) != LORA_TARGETS:
        raise ComparisonEvidenceError(f"{label}: LoRA target modules differ from the lock")

    optimization = _mapping(
        training.get("optimization"), label, "provenance.training_run.optimization"
    )
    numeric_locks = {
        "learning_rate": 0.0001,
        "epochs_requested": 6,
        "batch_size": arm_lock["batch_size"],
        "gradient_accumulation": arm_lock["gradient_accumulation"],
        "effective_batch_size": 32,
        "max_length": arm_lock["max_length"],
        "first_supervised_token_weight": 3.0,
        "warmup_ratio": 0.05,
        "warmup_steps": 2,
        "weight_decay": 0.01,
        "early_stopping_patience": 2,
        "max_grad_norm": 1.0,
        "per_device_eval_batch_size": arm_lock["eval_batch_size"],
    }
    for key, expected in numeric_locks.items():
        _expect_number(optimization, key, expected, label, "training.optimization")
    text_locks = {
        "loss_mode": "per_example_completion_mean",
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
    }
    boolean_locks = {
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "full_determinism": True,
    }
    if any(optimization.get(key) != expected for key, expected in text_locks.items()):
        raise ComparisonEvidenceError(f"{label}: optimizer/scheduler identity is not locked")
    if any(optimization.get(key) is not expected for key, expected in boolean_locks.items()):
        raise ComparisonEvidenceError(f"{label}: deterministic training identity is not locked")

    identity_payload = {
        key: training[key]
        for key in ("seed", "datasets", "base_model", "prompt", "loss", "lora", "optimization")
    }
    identity_sha256 = _sha256(
        training.get("identity_sha256"),
        label,
        "provenance.training_run.identity_sha256",
    )
    if identity_sha256 != _json_identity(identity_payload):
        raise ComparisonEvidenceError(f"{label}: training identity hash is inconsistent")
    settings_identity = _json_identity(
        {key: training[key] for key in ("prompt", "loss", "lora", "optimization")}
    )
    model_lock = _safe_file_fingerprint(
        training.get("model_lock"),
        label,
        "provenance.training_run.model_lock",
        filename="model.lock.json",
    )
    adapter_identity = _training_adapter_artifact(
        training.get("adapter_artifact"),
        label,
    )
    checkpoint = _checkpoint_selection(
        training.get("checkpoint_selection"),
        label,
    )
    trainer_code = _sha_mapping(
        training.get("trainer_code_sha256"),
        label,
        "provenance.training_run.trainer_code_sha256",
    )
    artifact_binding = _sha256(
        training.get("artifact_binding_sha256"),
        label,
        "provenance.training_run.artifact_binding_sha256",
    )
    binding_payload = {
        "format": "lfm25_training_adapter_binding_v1",
        "training_identity_sha256": identity_sha256,
        "model_lock": model_lock,
        "adapter_artifact_identity_sha256": adapter_identity,
        "checkpoint_selection": checkpoint,
        "trainer_code_sha256": trainer_code,
    }
    if artifact_binding != _json_identity(binding_payload):
        raise ComparisonEvidenceError(f"{label}: training artifact binding hash is inconsistent")
    return {
        "manifest_sha256": manifest_sha256,
        "identity_sha256": identity_sha256,
        "train_sha256": train_sha256,
        "eval_sha256": eval_sha256,
        "data_report": data_report,
        "base_model_files": base_files,
        "base_model_identity_sha256": base_identity,
        "contract_sha256": contract_sha256,
        "settings_identity_sha256": settings_identity,
        "model_lock": model_lock,
        "adapter_identity": adapter_identity,
        "checkpoint_selection": checkpoint,
        "trainer_code_sha256": trainer_code,
        "artifact_binding_sha256": artifact_binding,
    }


def _selector_acceptance(
    metrics: Mapping[str, Any],
    invocations: int,
    label: str,
) -> dict[str, int | float]:
    block = _mapping(
        metrics.get("candidate_protocol_acceptance"),
        label,
        "candidate_protocol_acceptance",
    )
    integers = {
        key: _integer(block.get(key), label, f"candidate_protocol_acceptance.{key}")
        for key in (
            "model_invocations",
            "accepted_outputs",
            "rejected_outputs",
            "accepted_transactions",
            "source_grounded_transactions",
        )
    }
    if integers["model_invocations"] != invocations:
        raise ComparisonEvidenceError(f"{label}: acceptance invocation count disagrees")
    if integers["accepted_outputs"] + integers["rejected_outputs"] != invocations:
        raise ComparisonEvidenceError(f"{label}: acceptance counts do not partition invocations")
    if integers["accepted_transactions"] > integers["accepted_outputs"]:
        raise ComparisonEvidenceError(f"{label}: accepted transaction count is impossible")
    if integers["source_grounded_transactions"] > integers["accepted_transactions"]:
        raise ComparisonEvidenceError(f"{label}: grounded transaction count is impossible")
    acceptance_rate = _number(
        block.get("strict_schema_acceptance_rate"),
        label,
        "candidate_protocol_acceptance.strict_schema_acceptance_rate",
    )
    grounded_rate = _number(
        block.get("source_grounded_transaction_rate"),
        label,
        "candidate_protocol_acceptance.source_grounded_transaction_rate",
    )
    expected_acceptance = Decimal(integers["accepted_outputs"]) / Decimal(invocations)
    if abs(acceptance_rate - expected_acceptance) > Decimal("0.000001"):
        raise ComparisonEvidenceError(f"{label}: strict acceptance rate disagrees with counts")
    accepted_transactions = integers["accepted_transactions"]
    if accepted_transactions <= 0:
        raise ComparisonEvidenceError(f"{label}: no accepted transactions ground the source check")
    expected_grounded = Decimal(integers["source_grounded_transactions"]) / Decimal(
        accepted_transactions
    )
    if abs(grounded_rate - expected_grounded) > Decimal("0.000001"):
        raise ComparisonEvidenceError(f"{label}: source-grounded rate disagrees with counts")

    reason_counts = _mapping(metrics.get("selector_reason_counts"), label, "selector_reason_counts")
    normalized_reasons = {
        str(reason): _integer(count, label, f"selector_reason_counts.{reason}")
        for reason, count in reason_counts.items()
        if isinstance(reason, str) and reason
    }
    if sum(normalized_reasons.values()) != invocations:
        raise ComparisonEvidenceError(f"{label}: selector reasons do not partition invocations")
    accepted_reason_count = normalized_reasons.get(
        "accepted_not_transaction", 0
    ) + normalized_reasons.get("accepted_transaction", 0)
    if accepted_reason_count != integers["accepted_outputs"]:
        raise ComparisonEvidenceError(f"{label}: accepted reason counts disagree")
    if normalized_reasons.get("accepted_transaction", 0) != accepted_transactions:
        raise ComparisonEvidenceError(f"{label}: accepted transaction reasons disagree")

    statuses = _mapping(metrics.get("selector_status_counts"), label, "selector_status_counts")
    normalized_statuses = {
        str(status): _integer(count, label, f"selector_status_counts.{status}")
        for status, count in statuses.items()
        if isinstance(status, str) and status
    }
    if set(normalized_statuses) - {"transaction", "null", "invalid"}:
        raise ComparisonEvidenceError(f"{label}: selector status vocabulary is not V1")
    if sum(normalized_statuses.values()) != invocations:
        raise ComparisonEvidenceError(f"{label}: selector statuses do not partition invocations")
    if normalized_statuses.get("transaction", 0) != accepted_transactions:
        raise ComparisonEvidenceError(f"{label}: selector transaction statuses disagree")
    if normalized_statuses.get("invalid", 0) != integers["rejected_outputs"]:
        raise ComparisonEvidenceError(f"{label}: selector rejection statuses disagree")

    conditional = _mapping(metrics.get("conditional_model"), label, "conditional_model")
    conditional_counts = _mapping(conditional.get("counts"), label, "conditional_model.counts")
    if (
        _integer(conditional_counts.get("rows"), label, "conditional_model.counts.rows")
        != invocations
    ):
        raise ComparisonEvidenceError(f"{label}: conditional model row count disagrees")
    schema_valid = _integer(
        conditional_counts.get("schema_valid"),
        label,
        "conditional_model.counts.schema_valid",
    )
    if schema_valid != integers["accepted_outputs"]:
        raise ComparisonEvidenceError(f"{label}: strict parser and schema counts disagree")

    return {
        **integers,
        "strict_schema_acceptance_rate": float(acceptance_rate),
        "source_grounded_transaction_rate": float(grounded_rate),
    }


def _extract_run(
    metrics: Mapping[str, Any],
    metrics_sha256: str,
    arm: str,
    seed: int,
) -> _RunEvidence:
    label = f"{arm} seed {seed}"
    counts = _mapping(metrics.get("counts"), label, "counts")
    rows = _integer(counts.get("rows"), label, "counts.rows", minimum=1)
    transaction_exact = _counter_integer(counts, "transaction_exact", label)
    fp = _counter_integer(counts, "fp", label)
    runtime = _mapping(metrics.get("runtime"), label, "runtime")
    if _integer(runtime.get("rows"), label, "runtime.rows", minimum=1) != rows:
        raise ComparisonEvidenceError(f"{label}: runtime and score row counts disagree")
    invocations = _integer(
        runtime.get("model_invocations"), label, "runtime.model_invocations", minimum=1
    )
    if invocations > rows:
        raise ComparisonEvidenceError(f"{label}: model invocations exceed rows")
    evaluation_batch_size = _integer(
        runtime.get("batch_size"), label, "runtime.batch_size", minimum=1
    )
    if evaluation_batch_size != EVALUATION_LOCKS[arm]["batch_size"]:
        raise ComparisonEvidenceError(f"{label}: evaluation batch size differs from the lock")

    provenance = _mapping(metrics.get("provenance"), label, "provenance")
    prefilter = _prefilter_evidence(metrics, runtime, provenance, rows, invocations, label)
    expected_transaction_exact = _score_rate_count(
        metrics.get("transaction_only_exact_match"),
        prefilter["gold_transactions"],
        label,
        "transaction_only_exact_match",
    )
    expected_fp = _score_rate_count(
        metrics.get("conditional_ghost_rate"),
        prefilter["gold_nulls"],
        label,
        "conditional_ghost_rate",
    )
    if transaction_exact != expected_transaction_exact:
        raise ComparisonEvidenceError(
            f"{label}: transaction_only_exact_match disagrees with counts"
        )
    if fp != expected_fp:
        raise ComparisonEvidenceError(f"{label}: conditional_ghost_rate disagrees with counts")

    dataset = _mapping(provenance.get("dataset"), label, "provenance.dataset")
    dataset_sha256 = _sha256(dataset.get("sha256"), label, "provenance.dataset.sha256")
    dataset_bytes = _integer(dataset.get("bytes"), label, "provenance.dataset.bytes", minimum=1)
    dataset_rows = _integer(
        dataset.get("row_count"),
        label,
        "provenance.dataset.row_count",
        minimum=1,
    )
    if (
        dataset_rows != rows
        or dataset_rows != DIAGNOSTIC_ROWS
        or dataset.get("row_limit") is not None
    ):
        raise ComparisonEvidenceError(
            f"{label}: dataset evidence is not the complete locked diagnostic set"
        )
    model_files = _fingerprinted_files(provenance.get("model"), label, "provenance.model")
    model_lock = _safe_file_fingerprint(
        provenance.get("model_lock"),
        label,
        "provenance.model_lock",
        filename="model.lock.json",
    )
    adapter_identity = _adapter_identity(provenance.get("adapter"), label)
    evaluator_sha256 = _fingerprint_sha(provenance.get("evaluator"), label, "provenance.evaluator")
    code_sha256 = _code_identity(provenance.get("code_sha256"), label)
    decode = _decode_evidence(provenance.get("decode"), runtime, arm, seed, label)

    oracle = None
    acceptance = None
    protocol = None
    generation_engine_sha256 = None
    candidate_profile = None
    platform_gates = None
    if arm == "direct":
        if runtime.get("profile") != "pocketfinancer":
            raise ComparisonEvidenceError(f"{label}: direct profile is not PocketFinancer")
        profile = _mapping(provenance.get("profile"), label, "provenance.profile")
        if profile.get("name") != "pocketfinancer_hf_training_evaluation":
            raise ComparisonEvidenceError(f"{label}: direct HF provenance identity is wrong")
        prefilter_contract = dict(
            _mapping(
                profile.get("android_current_prompt_contract"),
                label,
                "provenance.profile.android_current_prompt_contract",
            )
        )
    else:
        if provenance.get("pipeline") != "pocketfinancer_candidate_protocol_v1_hf":
            raise ComparisonEvidenceError(f"{label}: selector HF pipeline identity is wrong")
        if runtime.get("model_output_protocol") != PROTOCOL_NAME:
            raise ComparisonEvidenceError(f"{label}: selector runtime protocol identity is wrong")
        hybrid = _mapping(metrics.get("hybrid_safety"), label, "hybrid_safety")
        if hybrid.get("enabled") is not False:
            raise ComparisonEvidenceError(f"{label}: hybrid safety changes the controlled arm")
        prefilter_contract = dict(
            _mapping(
                provenance.get("prefilter_contract"),
                label,
                "provenance.prefilter_contract",
            )
        )
        oracle = _oracle_coverage(metrics.get("candidate_oracle"), label)
        protocol = _protocol_provenance(provenance.get("candidate_protocol"), provenance, label)
        acceptance = _selector_acceptance(metrics, invocations, label)
        generation_engine_sha256 = _fingerprint_sha(
            provenance.get("generation_engine"),
            label,
            "provenance.generation_engine",
        )
        candidate_profile_value = _mapping(
            provenance.get("candidate_profile"),
            label,
            "provenance.candidate_profile",
        )
        if set(candidate_profile_value) != {
            "candidate",
            "baseline",
            "golden_vectors",
        }:
            raise _fail(label, "provenance.candidate_profile")
        candidate_profile = {
            "candidate": _safe_file_fingerprint(
                candidate_profile_value.get("candidate"),
                label,
                "provenance.candidate_profile.candidate",
                filename="pocketfinancer-candidate-v1.json",
            ),
            "baseline": _safe_file_fingerprint(
                candidate_profile_value.get("baseline"),
                label,
                "provenance.candidate_profile.baseline",
                filename="pocketfinancer-android-current.json",
            ),
            "golden_vectors": _safe_file_fingerprint(
                candidate_profile_value.get("golden_vectors"),
                label,
                "provenance.candidate_profile.golden_vectors",
                filename="candidate_protocol_v1_golden.json",
            ),
        }
        platform_gates = dict(
            _mapping(provenance.get("platform_gates"), label, "provenance.platform_gates")
        )
        if platform_gates != PLATFORM_GATES:
            raise ComparisonEvidenceError(f"{label}: platform gates are not fail-closed")

    if arm == "direct":
        expected_training_contract = _json_identity(prefilter_contract)
    else:
        assert protocol is not None
        expected_training_contract = _json_identity({"profile": PROTOCOL_NAME, **protocol})
    training = _training_run_evidence(
        provenance.get("training_run"),
        arm=arm,
        seed=seed,
        model_files=model_files,
        expected_contract_sha256=expected_training_contract,
        label=label,
    )
    if training["model_lock"] != model_lock:
        raise ComparisonEvidenceError(
            f"{label}: training and evaluation model-lock fingerprints differ"
        )
    if training["adapter_identity"] != adapter_identity:
        raise ComparisonEvidenceError(
            f"{label}: training manifest is not bound to the evaluated adapter"
        )

    return _RunEvidence(
        arm=arm,
        seed=seed,
        metrics_sha256=metrics_sha256,
        rows=rows,
        model_invocations=invocations,
        transaction_exact=transaction_exact,
        fp=fp,
        evaluation_batch_size=evaluation_batch_size,
        prefilter=prefilter,
        training=training,
        dataset_sha256=dataset_sha256,
        dataset_bytes=dataset_bytes,
        model_files=model_files,
        model_lock=model_lock,
        adapter_identity=adapter_identity,
        evaluator_sha256=evaluator_sha256,
        generation_engine_sha256=generation_engine_sha256,
        code_sha256=code_sha256,
        decode=decode,
        prefilter_contract=prefilter_contract,
        oracle=oracle,
        acceptance=acceptance,
        protocol_provenance=protocol,
        candidate_profile=candidate_profile,
        platform_gates=platform_gates,
    )


def _require_exact_seed_mapping(
    values: Mapping[int, Any],
    label: str,
) -> None:
    if set(values) != set(SEEDS):
        raise ComparisonEvidenceError(f"{label}: evidence must contain exact seeds 17, 29, and 43")


def _normalize_trusted_anchors(value: Any) -> dict[str, Any]:
    label = "trusted anchors"
    anchors = _mapping(value, label, "root")
    if anchors.get("schema_version") != 1:
        raise _fail(label, "schema_version")

    diagnostic_value = _mapping(
        anchors.get("diagnostic_dataset"),
        label,
        "diagnostic_dataset",
    )
    diagnostic = {
        **_safe_file_fingerprint(
            diagnostic_value,
            label,
            "diagnostic_dataset",
            filename="extraction_ds.jsonl",
        ),
        "rows": _integer(
            diagnostic_value.get("rows"),
            label,
            "diagnostic_dataset.rows",
            minimum=1,
        ),
        "row_limit": diagnostic_value.get("row_limit"),
    }
    if diagnostic["rows"] != DIAGNOSTIC_ROWS or diagnostic["row_limit"] is not None:
        raise ComparisonEvidenceError(
            "trusted anchors: diagnostic dataset is not the complete locked 203 rows"
        )
    diagnostic_prefilter = _prefilter_summary(
        anchors.get("diagnostic_prefilter"),
        label,
        "diagnostic_prefilter",
    )
    if diagnostic_prefilter["n"] != diagnostic["rows"]:
        raise ComparisonEvidenceError(
            "trusted anchors: diagnostic prefilter row count differs from the dataset"
        )

    candidate_data_value = _mapping(
        anchors.get("candidate_data"),
        label,
        "candidate_data",
    )
    candidate_rows = _mapping(
        candidate_data_value.get("rows"),
        label,
        "candidate_data.rows",
    )
    if candidate_rows != {"train": 152, "dev": 29}:
        raise _fail(label, "candidate_data.rows")
    candidate_data = {
        "report": _safe_file_fingerprint(
            candidate_data_value.get("report"),
            label,
            "candidate_data.report",
            filename="candidate_protocol_v1_report.json",
        ),
        "train_sha256": _sha256(
            candidate_data_value.get("train_sha256"),
            label,
            "candidate_data.train_sha256",
        ),
        "dev_sha256": _sha256(
            candidate_data_value.get("dev_sha256"),
            label,
            "candidate_data.dev_sha256",
        ),
        "rows": {"train": 152, "dev": 29},
    }

    model_value = _mapping(anchors.get("model"), label, "model")
    if model_value.get("id") != "LiquidAI/LFM2.5-350M":
        raise _fail(label, "model.id")
    revision = model_value.get("revision")
    if revision != MODEL_REVISION:
        raise _fail(label, "model.revision")
    model_files = _sha_mapping(model_value.get("files"), label, "model.files")
    if (
        "config.json" not in model_files
        or not any(name.endswith((".safetensors", ".bin")) for name in model_files)
        or not ({"tokenizer.json", "tokenizer.model"} & set(model_files))
    ):
        raise _fail(label, "model.files")
    model = {
        "lock": _safe_file_fingerprint(
            model_value.get("lock"),
            label,
            "model.lock",
            filename="model.lock.json",
        ),
        "id": "LiquidAI/LFM2.5-350M",
        "revision": revision,
        "files": model_files,
    }

    profile_values = _mapping(anchors.get("profiles"), label, "profiles")
    profiles = {
        key: _safe_file_fingerprint(
            profile_values.get(key),
            label,
            f"profiles.{key}",
            filename=filename,
        )
        for key, filename in (
            ("candidate", "pocketfinancer-candidate-v1.json"),
            ("baseline", "pocketfinancer-android-current.json"),
            ("golden_vectors", "candidate_protocol_v1_golden.json"),
        )
    }
    prefilter_contract = dict(
        _mapping(anchors.get("prefilter_contract"), label, "prefilter_contract")
    )
    candidate_protocol = dict(
        _mapping(anchors.get("candidate_protocol"), label, "candidate_protocol")
    )
    if not prefilter_contract or not candidate_protocol:
        raise _fail(label, "protocol contracts")

    evaluator_code = _sha_mapping(
        anchors.get("evaluator_code_sha256"),
        label,
        "evaluator_code_sha256",
    )
    required_evaluators = {
        "direct",
        "selector",
        "selector_generation_engine",
        "comparator_module",
        "comparator_cli",
    }
    if set(evaluator_code) != required_evaluators:
        raise _fail(label, "evaluator_code_sha256")
    repo_root = Path(__file__).resolve().parent.parent
    current_comparator_code = {
        "comparator_module": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparator_cli": hashlib.sha256(
            (repo_root / "scripts" / "compare_lfm25_candidate_protocol_v1.py").read_bytes()
        ).hexdigest(),
    }
    if any(evaluator_code[name] != digest for name, digest in current_comparator_code.items()):
        raise ComparisonEvidenceError(
            "trusted anchors: comparator code differs from the current checkout"
        )
    trainer_values = _mapping(
        anchors.get("trainer_code_sha256"),
        label,
        "trainer_code_sha256",
    )
    if set(trainer_values) != {"direct", "selector"}:
        raise _fail(label, "trainer_code_sha256")
    trainers = {
        arm: _sha_mapping(
            trainer_values[arm],
            label,
            f"trainer_code_sha256.{arm}",
        )
        for arm in ("direct", "selector")
    }
    platform = dict(_mapping(anchors.get("platform_gates"), label, "platform_gates"))
    if platform != PLATFORM_GATES:
        raise ComparisonEvidenceError("trusted anchors: platform gates are not fail-closed")
    return {
        "schema_version": 1,
        "diagnostic_dataset": diagnostic,
        "diagnostic_prefilter": diagnostic_prefilter,
        "candidate_data": candidate_data,
        "model": model,
        "profiles": profiles,
        "prefilter_contract": prefilter_contract,
        "candidate_protocol": candidate_protocol,
        "shared_code_sha256": _sha_mapping(
            anchors.get("shared_code_sha256"),
            label,
            "shared_code_sha256",
        ),
        "evaluator_code_sha256": evaluator_code,
        "trainer_code_sha256": trainers,
        "platform_gates": dict(PLATFORM_GATES),
    }


def _require_common_identity(runs: list[_RunEvidence]) -> dict[str, Any]:
    dataset_identities = {(run.dataset_sha256, run.dataset_bytes) for run in runs}
    if len(dataset_identities) != 1:
        raise ComparisonEvidenceError("controlled matrix: dataset fingerprints differ")
    if any(run.rows != DIAGNOSTIC_ROWS for run in runs):
        raise ComparisonEvidenceError("controlled matrix: diagnostic row count is not 203")
    if len({run.model_invocations for run in runs}) != 1:
        raise ComparisonEvidenceError("controlled matrix: prefilter/model invocation counts differ")
    if len({_json_identity(run.code_sha256) for run in runs}) != 1:
        raise ComparisonEvidenceError("controlled matrix: code provenance differs")
    if len({_json_identity(run.prefilter_contract) for run in runs}) != 1:
        raise ComparisonEvidenceError("controlled matrix: prefilter contract provenance differs")

    prefilter_identities = {_json_identity(run.prefilter) for run in runs}
    if len(prefilter_identities) != 1:
        raise ComparisonEvidenceError("controlled matrix: prefilter aggregates differ")
    training_data = {(run.training["train_sha256"], run.training["eval_sha256"]) for run in runs}
    if len(training_data) != 1:
        raise ComparisonEvidenceError("controlled matrix: training train/dev fingerprints differ")
    if len({_json_identity(run.training["data_report"]) for run in runs}) != 1:
        raise ComparisonEvidenceError("controlled matrix: candidate data reports differ")
    if len({_json_identity(run.model_lock) for run in runs}) != 1:
        raise ComparisonEvidenceError("controlled matrix: model-lock fingerprints differ")
    training_base_identities = {run.training["base_model_identity_sha256"] for run in runs}
    if len(training_base_identities) != 1:
        raise ComparisonEvidenceError("controlled matrix: training base-model identities differ")
    manifest_hashes = [run.training["manifest_sha256"] for run in runs]
    if len(set(manifest_hashes)) != len(runs):
        raise ComparisonEvidenceError(
            "controlled matrix: a training manifest fingerprint is reused across runs"
        )
    training_run_identities = [run.training["identity_sha256"] for run in runs]
    if len(set(training_run_identities)) != len(runs):
        raise ComparisonEvidenceError(
            "controlled matrix: a training run identity is reused across runs"
        )
    artifact_bindings = [run.training["artifact_binding_sha256"] for run in runs]
    if len(set(artifact_bindings)) != len(runs):
        raise ComparisonEvidenceError(
            "controlled matrix: a training artifact binding is reused across runs"
        )

    common_model_names = set.intersection(*(set(run.model_files) for run in runs))
    weight_names = {name for name in common_model_names if name.endswith((".safetensors", ".bin"))}
    tokenizer_names = common_model_names & {"tokenizer.json", "tokenizer.model"}
    if "config.json" not in common_model_names or not weight_names or not tokenizer_names:
        raise ComparisonEvidenceError("controlled matrix: insufficient common model fingerprints")
    common_model = {name: runs[0].model_files[name] for name in sorted(common_model_names)}
    for run in runs[1:]:
        if any(run.model_files[name] != digest for name, digest in common_model.items()):
            raise ComparisonEvidenceError("controlled matrix: base model fingerprints differ")
    if len({run.adapter_identity for run in runs}) != len(runs):
        raise ComparisonEvidenceError(
            "controlled matrix: an adapter identity is reused across runs"
        )

    for arm in ("direct", "selector"):
        arm_runs = [run for run in runs if run.arm == arm]
        if len({run.evaluator_sha256 for run in arm_runs}) != 1:
            raise ComparisonEvidenceError(f"controlled matrix: {arm} evaluator identity differs")
        if len({_json_identity(run.decode) for run in arm_runs}) != 1:
            raise ComparisonEvidenceError(f"controlled matrix: {arm} decode settings differ")
        if len({run.training["settings_identity_sha256"] for run in arm_runs}) != 1:
            raise ComparisonEvidenceError(
                f"controlled matrix: {arm} training settings identity differs"
            )
        if len({_json_identity(run.training["trainer_code_sha256"]) for run in arm_runs}) != 1:
            raise ComparisonEvidenceError(f"controlled matrix: {arm} trainer code identity differs")
        if len({run.training["contract_sha256"] for run in arm_runs}) != 1:
            raise ComparisonEvidenceError(
                f"controlled matrix: {arm} training prompt contract identity differs"
            )
    selectors = [run for run in runs if run.arm == "selector"]
    if len({_json_identity(run.protocol_provenance) for run in selectors}) != 1:
        raise ComparisonEvidenceError("controlled matrix: Candidate Protocol provenance differs")
    if len({_json_identity(run.oracle) for run in selectors}) != 1:
        raise ComparisonEvidenceError("controlled matrix: candidate oracle coverage differs")
    if len({run.generation_engine_sha256 for run in selectors}) != 1:
        raise ComparisonEvidenceError("controlled matrix: selector generation engines differ")
    if len({_json_identity(run.candidate_profile) for run in selectors}) != 1:
        raise ComparisonEvidenceError("controlled matrix: candidate profile anchors differ")
    if len({_json_identity(run.platform_gates) for run in selectors}) != 1:
        raise ComparisonEvidenceError("controlled matrix: platform gates differ")

    dataset_sha256, _dataset_bytes = next(iter(dataset_identities))
    train_sha256, eval_sha256 = next(iter(training_data))
    return {
        "dataset_sha256": dataset_sha256,
        "model_identity_sha256": _json_identity(common_model),
        "code_identity_sha256": _json_identity(runs[0].code_sha256),
        "prefilter_contract_identity_sha256": _json_identity(runs[0].prefilter_contract),
        "prefilter_identity_sha256": next(iter(prefilter_identities)),
        "candidate_protocol_identity_sha256": _json_identity(selectors[0].protocol_provenance),
        "training_data": {
            "train_sha256": train_sha256,
            "dev_sha256": eval_sha256,
        },
        "training_base_model_identity_sha256": next(iter(training_base_identities)),
        "training_settings_identity_sha256": {
            arm: next(run.training["settings_identity_sha256"] for run in runs if run.arm == arm)
            for arm in ("direct", "selector")
        },
        "training_manifest_sha256": {
            arm: {str(run.seed): run.training["manifest_sha256"] for run in runs if run.arm == arm}
            for arm in ("direct", "selector")
        },
        "training_run_identity_sha256": {
            arm: {str(run.seed): run.training["identity_sha256"] for run in runs if run.arm == arm}
            for arm in ("direct", "selector")
        },
    }


def _require_trusted_anchors(
    runs: list[_RunEvidence],
    value: Any,
) -> dict[str, Any]:
    anchors = _normalize_trusted_anchors(value)
    dataset = anchors["diagnostic_dataset"]
    diagnostic_prefilter = anchors["diagnostic_prefilter"]
    candidate_data = anchors["candidate_data"]
    model = anchors["model"]
    evaluators = anchors["evaluator_code_sha256"]
    trainers = anchors["trainer_code_sha256"]
    for run in runs:
        label = f"{run.arm} seed {run.seed}"
        if run.dataset_sha256 != dataset["sha256"] or run.dataset_bytes != dataset["bytes"]:
            raise ComparisonEvidenceError(
                f"{label}: diagnostic dataset differs from the live trusted anchor"
            )
        if run.prefilter != diagnostic_prefilter:
            raise ComparisonEvidenceError(
                f"{label}: prefilter aggregates differ from the live diagnostic anchor"
            )
        if (
            run.training["train_sha256"] != candidate_data["train_sha256"]
            or run.training["eval_sha256"] != candidate_data["dev_sha256"]
            or run.training["data_report"] != candidate_data["report"]
        ):
            raise ComparisonEvidenceError(
                f"{label}: training data differs from the verified live data report"
            )
        if (
            run.model_files != model["files"]
            or run.training["base_model_files"] != model["files"]
            or run.model_lock != model["lock"]
        ):
            raise ComparisonEvidenceError(
                f"{label}: model evidence differs from the live immutable model lock"
            )
        if run.prefilter_contract != anchors["prefilter_contract"]:
            raise ComparisonEvidenceError(
                f"{label}: prefilter contract differs from the live baseline profile"
            )
        if run.code_sha256 != anchors["shared_code_sha256"]:
            raise ComparisonEvidenceError(
                f"{label}: shared evaluator code differs from the current checkout"
            )
        if run.evaluator_sha256 != evaluators[run.arm]:
            raise ComparisonEvidenceError(f"{label}: evaluator differs from the current checkout")
        if run.training["trainer_code_sha256"] != trainers[run.arm]:
            raise ComparisonEvidenceError(
                f"{label}: trainer code differs from the current checkout"
            )
        if run.training["adapter_identity"] != run.adapter_identity:
            raise ComparisonEvidenceError(
                f"{label}: training manifest is not bound to the evaluated adapter"
            )
        if run.arm == "selector":
            if run.generation_engine_sha256 != evaluators["selector_generation_engine"]:
                raise ComparisonEvidenceError(
                    f"{label}: selector generation engine differs from the current checkout"
                )
            if run.protocol_provenance != anchors["candidate_protocol"]:
                raise ComparisonEvidenceError(
                    f"{label}: Candidate Protocol differs from the live trusted profile"
                )
            if run.candidate_profile != anchors["profiles"]:
                raise ComparisonEvidenceError(
                    f"{label}: candidate profile or golden-vector anchor differs"
                )
            if run.platform_gates != anchors["platform_gates"]:
                raise ComparisonEvidenceError(
                    f"{label}: runtime/device platform gates are not fail-closed"
                )

    return {
        "identity_sha256": _json_identity(anchors),
        "diagnostic_dataset": dict(dataset),
        "diagnostic_prefilter": dict(diagnostic_prefilter),
        "candidate_data": {
            "report": dict(candidate_data["report"]),
            "train_sha256": candidate_data["train_sha256"],
            "dev_sha256": candidate_data["dev_sha256"],
            "rows": dict(candidate_data["rows"]),
        },
        "model": {
            "lock": dict(model["lock"]),
            "id": model["id"],
            "revision": model["revision"],
            "files_identity_sha256": _json_identity(model["files"]),
        },
        "profiles": {key: dict(fingerprint) for key, fingerprint in anchors["profiles"].items()},
        "prefilter_contract_identity_sha256": _json_identity(anchors["prefilter_contract"]),
        "candidate_protocol_identity_sha256": _json_identity(anchors["candidate_protocol"]),
        "shared_code_identity_sha256": _json_identity(anchors["shared_code_sha256"]),
        "evaluator_code_sha256": dict(evaluators),
        "trainer_code_identity_sha256": {
            arm: _json_identity(trainers[arm]) for arm in ("direct", "selector")
        },
        "platform_gates": dict(anchors["platform_gates"]),
    }


def _compare_metrics(
    direct: Mapping[int, Mapping[str, Any]],
    selector: Mapping[int, Mapping[str, Any]],
    *,
    metrics_hashes: Mapping[str, Mapping[int, str]],
    trusted_anchors: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate six aggregate metric objects and assemble a privacy-safe report."""

    _require_exact_seed_mapping(direct, "direct")
    _require_exact_seed_mapping(selector, "selector")
    hashes = metrics_hashes
    if set(hashes) != {"direct", "selector"}:
        raise ComparisonEvidenceError("metrics hashes must identify direct and selector arms")
    _require_exact_seed_mapping(hashes["direct"], "direct metrics hashes")
    _require_exact_seed_mapping(hashes["selector"], "selector metrics hashes")
    hashes = {
        arm: {
            seed: _sha256(hashes[arm][seed], f"{arm} seed {seed}", "metrics sha256")
            for seed in SEEDS
        }
        for arm in ("direct", "selector")
    }

    direct_runs = {
        seed: _extract_run(direct[seed], hashes["direct"][seed], "direct", seed) for seed in SEEDS
    }
    selector_runs = {
        seed: _extract_run(selector[seed], hashes["selector"][seed], "selector", seed)
        for seed in SEEDS
    }
    runs = [*direct_runs.values(), *selector_runs.values()]
    identities = _require_common_identity(runs)
    trusted_evidence = (
        _require_trusted_anchors(runs, trusted_anchors) if trusted_anchors is not None else None
    )
    evidence_validated = trusted_evidence is not None

    per_seed: list[dict[str, Any]] = []
    transaction_checks: list[bool] = []
    ghost_checks: list[bool] = []
    schema_checks: list[bool] = []
    grounding_checks: list[bool] = []
    oracle_checks: list[bool] = []
    for seed in SEEDS:
        direct_run = direct_runs[seed]
        selector_run = selector_runs[seed]
        assert selector_run.acceptance is not None
        assert selector_run.oracle is not None
        exact_passed = selector_run.transaction_exact > direct_run.transaction_exact
        fp_passed = selector_run.fp <= direct_run.fp
        schema_passed = (
            selector_run.acceptance["accepted_outputs"]
            == selector_run.acceptance["model_invocations"]
            and selector_run.acceptance["rejected_outputs"] == 0
            and selector_run.acceptance["strict_schema_acceptance_rate"] == 1.0
        )
        grounding_passed = (
            selector_run.acceptance["accepted_transactions"]
            == selector_run.acceptance["source_grounded_transactions"]
            and selector_run.acceptance["source_grounded_transaction_rate"] == 1.0
        )
        oracle_passed = all(
            selector_run.oracle[field]["covered"] >= floor_covered
            and selector_run.oracle[field]["total"] == floor_total
            for field, (floor_covered, floor_total) in ORACLE_FLOORS.items()
        )
        transaction_checks.append(exact_passed)
        ghost_checks.append(fp_passed)
        schema_checks.append(schema_passed)
        grounding_checks.append(grounding_passed)
        oracle_checks.append(oracle_passed)
        per_seed.append(
            {
                "seed": seed,
                "direct": {
                    "transaction_exact": direct_run.transaction_exact,
                    "fp": direct_run.fp,
                },
                "selector": {
                    "transaction_exact": selector_run.transaction_exact,
                    "fp": selector_run.fp,
                    "strict_schema_acceptance_rate": selector_run.acceptance[
                        "strict_schema_acceptance_rate"
                    ],
                    "source_grounded_transaction_rate": selector_run.acceptance[
                        "source_grounded_transaction_rate"
                    ],
                },
                "delta": {
                    "transaction_exact": (
                        selector_run.transaction_exact - direct_run.transaction_exact
                    ),
                    "fp": selector_run.fp - direct_run.fp,
                },
                "checks": {
                    "transaction_exact_strictly_greater": exact_passed,
                    "fp_not_greater": fp_passed,
                    "strict_schema_acceptance_100_percent": schema_passed,
                    "source_grounded_accepted_transactions_100_percent": grounding_passed,
                    "oracle_coverage_floor": oracle_passed,
                },
            }
        )

    criteria = {
        "transaction_exact_strictly_greater_every_seed": all(transaction_checks),
        "fp_not_greater_every_seed": all(ghost_checks),
        "strict_schema_acceptance_100_percent_every_seed": all(schema_checks),
        "source_grounded_accepted_transactions_100_percent_every_seed": all(grounding_checks),
        "oracle_coverage_floor_every_seed": all(oracle_checks),
    }
    criteria_satisfied = all(criteria.values())
    controlled_passed = evidence_validated and criteria_satisfied
    selector_oracle = selector_runs[SEEDS[0]].oracle
    assert selector_oracle is not None
    hash_field = "metrics_file_sha256" if evidence_validated else "metrics_object_sha256"
    evidence_mode = "trusted_file_bytes" if evidence_validated else "in_memory_non_evidentiary"
    return {
        "schema_version": 1,
        "report_type": (
            "candidate_protocol_v1_controlled_hf_comparison"
            if evidence_validated
            else "candidate_protocol_v1_non_evidentiary_analysis"
        ),
        "protocol": PROTOCOL_NAME,
        "seeds": list(SEEDS),
        "evidence_mode": evidence_mode,
        "evidence": {
            **identities,
            hash_field: {
                arm: {str(seed): hashes[arm][seed] for seed in SEEDS}
                for arm in ("direct", "selector")
            },
            **({"trusted_anchors": trusted_evidence} if trusted_evidence else {}),
            "diagnostic_dataset": {
                "rows": DIAGNOSTIC_ROWS,
                "role": "locked_reused_diagnostic_only",
                "fresh_test": False,
            },
            "candidate_oracle": selector_oracle,
        },
        "per_seed": per_seed,
        "controlled_hf_gate": {
            "passed": controlled_passed,
            "evidence_validated": evidence_validated,
            "criteria_satisfied": criteria_satisfied,
            "criteria": criteria,
            "scope": "HF research evidence only",
            "reason": (
                None
                if evidence_validated
                else "In-memory metric objects are analysis-only and cannot satisfy the gate."
            ),
        },
        "product_promotion": {
            "allowed": False,
            "blocked": True,
            "reason": (
                "The 203-row dataset is a reused diagnostic and runtime/product gates remain unmet."
            ),
        },
        "unmet_gates": {
            "fresh_human_gold": {
                "status": "unmet",
                "required_rows": 1436,
                "dataset_role": "fresh_human_gold_template_sender_held_out",
            },
            "selector_gguf": {"status": "unmet"},
            "android_runtime": {"status": "unmet"},
            "ios_runtime": {
                "status": "unmet",
                "reason": (
                    "No iOS runtime implements or has been validated against Candidate Protocol V1."
                ),
            },
            "android_device": {"status": "unmet"},
            "ios_device": {
                "status": "unmet",
                "reason": "No instrumented iOS device evidence exists for Candidate Protocol V1.",
            },
        },
    }


def compare_metrics(
    direct: Mapping[int, Mapping[str, Any]],
    selector: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare in-memory objects without making controlled-evidence claims."""

    hashes = {
        "direct": {seed: _json_identity(direct[seed]) for seed in SEEDS},
        "selector": {seed: _json_identity(selector[seed]) for seed in SEEDS},
    }
    return _compare_metrics(
        direct,
        selector,
        metrics_hashes=hashes,
        trusted_anchors=None,
    )


def compare_metric_files(
    direct_paths: Mapping[int, Path],
    selector_paths: Mapping[int, Path],
    *,
    trusted_anchors: Mapping[str, Any],
) -> dict[str, Any]:
    """Read exactly six aggregate ``metrics.json`` files and compare them."""

    _require_exact_seed_mapping(direct_paths, "direct paths")
    _require_exact_seed_mapping(selector_paths, "selector paths")
    metrics: dict[str, dict[int, dict[str, Any]]] = {"direct": {}, "selector": {}}
    hashes: dict[str, dict[int, str]] = {"direct": {}, "selector": {}}
    for arm, paths in (("direct", direct_paths), ("selector", selector_paths)):
        for seed in SEEDS:
            value, digest = _read_metrics(Path(paths[seed]), f"{arm} seed {seed}")
            metrics[arm][seed] = value
            hashes[arm][seed] = digest
    return _compare_metrics(
        metrics["direct"],
        metrics["selector"],
        metrics_hashes=hashes,
        trusted_anchors=trusted_anchors,
    )


def write_report(report: Mapping[str, Any], output_path: Path, *, force: bool = False) -> None:
    """Atomically write an aggregate report, refusing replacement by default."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    reserved = False
    try:
        if not force:
            try:
                descriptor = os.open(
                    output_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite existing report: {output_path.name}"
                ) from exc
            else:
                os.close(descriptor)
                reserved = True
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, output_path)
        reserved = False
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if reserved and output_path.exists():
            output_path.unlink()
