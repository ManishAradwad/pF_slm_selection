"""Validate aggregate-only host and Android-device runtime evidence packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "RESULTS"
SYNTHETIC_FIXTURE = (
    REPOSITORY_ROOT
    / "tests/fixtures/pocketfinancer_android_runtime_evidence_synthetic.json"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "configs/contracts/pocketfinancer-android-runtime-evidence-v1.json"
)
BASELINE_PATH = (
    REPOSITORY_ROOT
    / "configs/baselines/pocketfinancer-android-552ffbdf-phase-c.json"
)
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
METRICS = (
    "prompt_eval_ms",
    "generation_ms",
    "generated_tokens",
    "tokens_per_second",
    "peak_rss_mib",
    "battery_delta_pct",
)


class AndroidRuntimeEvidenceError(ValueError):
    """Raised when runtime evidence is unsafe, malformed, or ambiguously labelled."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AndroidRuntimeEvidenceError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise AndroidRuntimeEvidenceError(
            f"{name} keys differ: missing={missing}, unexpected={unexpected}"
        )


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise AndroidRuntimeEvidenceError(
            f"{name} must be a lowercase opaque identifier"
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AndroidRuntimeEvidenceError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _contract() -> Mapping[str, Any]:
    return _mapping(
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8")),
        "contract",
    )


def _validate_metric(value: Any, name: str, total: int) -> None:
    if value is None:
        return
    metric = _mapping(value, name)
    _exact_keys(
        metric,
        {"count", "minimum", "p50", "p95", "maximum"},
        name,
    )
    count = _nonnegative_int(metric["count"], f"{name}.count")
    if count > total:
        raise AndroidRuntimeEvidenceError(f"{name}.count exceeds sample total")
    points: list[float] = []
    for key in ("minimum", "p50", "p95", "maximum"):
        raw = metric[key]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or raw < 0
        ):
            raise AndroidRuntimeEvidenceError(
                f"{name}.{key} must be nonnegative"
            )
        points.append(float(raw))
    if points != sorted(points):
        raise AndroidRuntimeEvidenceError(
            f"{name} quantiles are not monotonic"
        )
    if count == 0 and any(points):
        raise AndroidRuntimeEvidenceError(
            f"{name} has values with zero observations"
        )


def _validate_capture(
    value: Any,
    index: int,
    gap_codes: set[str],
) -> dict[str, Any]:
    capture = _mapping(value, f"captures[{index}]")
    _exact_keys(
        capture,
        {
            "capture_id",
            "evidence_class",
            "status",
            "provenance",
            "runtime",
            "sample",
            "measurements",
            "gaps",
        },
        f"captures[{index}]",
    )
    _identifier(capture["capture_id"], f"captures[{index}].capture_id")
    evidence_class = capture["evidence_class"]
    if evidence_class not in {"host", "android_device"}:
        raise AndroidRuntimeEvidenceError(
            "evidence_class must be host or android_device"
        )
    status = capture["status"]
    if status not in {"measured", "synthetic_fixture", "not_measured"}:
        raise AndroidRuntimeEvidenceError("unsupported capture status")

    provenance = _mapping(
        capture["provenance"],
        f"captures[{index}].provenance",
    )
    _exact_keys(
        provenance,
        {
            "environment_id",
            "os",
            "architecture",
            "engine",
            "device_fingerprint_sha256",
            "android_api_level",
        },
        f"captures[{index}].provenance",
    )
    _identifier(provenance["environment_id"], "environment_id")
    _identifier(provenance["os"], "os")
    _identifier(provenance["architecture"], "architecture")
    _identifier(provenance["engine"], "engine")
    fingerprint = provenance["device_fingerprint_sha256"]
    api_level = provenance["android_api_level"]
    if evidence_class == "host":
        if (
            fingerprint is not None
            or api_level is not None
            or provenance["os"] == "android"
        ):
            raise AndroidRuntimeEvidenceError(
                "host evidence contains Android-device provenance"
            )
    else:
        if provenance["os"] != "android":
            raise AndroidRuntimeEvidenceError(
                "Android-device evidence must declare Android OS"
            )
        if status == "not_measured":
            if fingerprint is not None or api_level is not None:
                raise AndroidRuntimeEvidenceError(
                    "an unmeasured device capture cannot claim device identity"
                )
        else:
            if (
                not isinstance(fingerprint, str)
                or SHA256_RE.fullmatch(fingerprint) is None
            ):
                raise AndroidRuntimeEvidenceError(
                    "device fingerprint must be SHA-256, never raw"
                )
            if (
                isinstance(api_level, bool)
                or not isinstance(api_level, int)
                or api_level <= 0
            ):
                raise AndroidRuntimeEvidenceError(
                    "Android API level must be positive"
                )

    runtime = _mapping(capture["runtime"], f"captures[{index}].runtime")
    _exact_keys(
        runtime,
        {
            "runtime_variant_id",
            "model_artifact_sha256",
            "chat_template",
            "grammar_enabled",
            "thinking_enabled",
            "thinking_token_budget",
            "answer_token_budget",
        },
        f"captures[{index}].runtime",
    )
    _identifier(runtime["runtime_variant_id"], "runtime_variant_id")
    artifact_hash = runtime["model_artifact_sha256"]
    if artifact_hash is not None and (
        not isinstance(artifact_hash, str)
        or SHA256_RE.fullmatch(artifact_hash) is None
    ):
        raise AndroidRuntimeEvidenceError(
            "model_artifact_sha256 must be null or SHA-256"
        )
    _identifier(runtime["chat_template"], "chat_template")
    if not isinstance(runtime["grammar_enabled"], bool):
        raise AndroidRuntimeEvidenceError("grammar_enabled must be boolean")
    if not isinstance(runtime["thinking_enabled"], bool):
        raise AndroidRuntimeEvidenceError("thinking_enabled must be boolean")
    _nonnegative_int(
        runtime["thinking_token_budget"],
        "thinking_token_budget",
    )
    if (
        _nonnegative_int(
            runtime["answer_token_budget"],
            "answer_token_budget",
        )
        <= 0
    ):
        raise AndroidRuntimeEvidenceError(
            "answer_token_budget must be positive"
        )

    sample = _mapping(capture["sample"], f"captures[{index}].sample")
    _exact_keys(
        sample,
        {
            "total",
            "success",
            "null",
            "error",
            "stopped",
            "cache_attempts",
            "cache_hits",
        },
        f"captures[{index}].sample",
    )
    counts = {
        key: _nonnegative_int(raw, f"sample.{key}")
        for key, raw in sample.items()
    }
    if (
        counts["success"]
        + counts["null"]
        + counts["error"]
        + counts["stopped"]
        != counts["total"]
    ):
        raise AndroidRuntimeEvidenceError(
            "sample outcomes must sum to total"
        )
    if not (
        counts["cache_hits"]
        <= counts["cache_attempts"]
        <= counts["total"]
    ):
        raise AndroidRuntimeEvidenceError("cache counts are inconsistent")

    measurements = _mapping(
        capture["measurements"],
        f"captures[{index}].measurements",
    )
    _exact_keys(
        measurements,
        set(METRICS),
        f"captures[{index}].measurements",
    )
    for metric_name in METRICS:
        _validate_metric(
            measurements[metric_name],
            metric_name,
            counts["total"],
        )

    gaps = capture["gaps"]
    if (
        not isinstance(gaps, list)
        or any(item not in gap_codes for item in gaps)
    ):
        raise AndroidRuntimeEvidenceError(
            "capture gaps must use frozen aggregate-safe codes"
        )
    if len(gaps) != len(set(gaps)):
        raise AndroidRuntimeEvidenceError(
            "capture gap codes must be unique"
        )
    if status == "not_measured":
        if counts["total"] != 0 or any(
            metric is not None for metric in measurements.values()
        ):
            raise AndroidRuntimeEvidenceError(
                "not_measured captures cannot contain measurements"
            )
        if not gaps:
            raise AndroidRuntimeEvidenceError(
                "not_measured captures must state a gap"
            )
    elif counts["total"] == 0:
        raise AndroidRuntimeEvidenceError(
            "measured and synthetic captures require observations"
        )

    return {
        "evidence_class": evidence_class,
        "status": status,
        "sample_total": counts["total"],
        "measured_fields": sum(
            metric is not None for metric in measurements.values()
        ),
    }


def validate_runtime_evidence_package(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a content-free aggregate package and return a safe summary."""

    _exact_keys(
        package,
        {
            "contract_id",
            "contract_version",
            "privacy_classification",
            "baseline",
            "captures",
        },
        "runtime evidence package",
    )
    contract = _contract()
    if package.get("contract_id") != contract["contract_id"]:
        raise AndroidRuntimeEvidenceError(
            "runtime evidence contract ID differs"
        )
    if package.get("contract_version") != contract["contract_version"]:
        raise AndroidRuntimeEvidenceError(
            "runtime evidence contract version differs"
        )
    if (
        package.get("privacy_classification")
        != contract["privacy_classification"]
    ):
        raise AndroidRuntimeEvidenceError(
            "runtime evidence privacy classification differs"
        )

    baseline = _mapping(package["baseline"], "baseline")
    _exact_keys(
        baseline,
        {
            "baseline_id",
            "baseline_manifest_sha256",
            "android_commit",
        },
        "baseline",
    )
    _identifier(baseline["baseline_id"], "baseline_id")
    if (
        SHA256_RE.fullmatch(
            str(baseline["baseline_manifest_sha256"])
        )
        is None
    ):
        raise AndroidRuntimeEvidenceError(
            "baseline_manifest_sha256 must be SHA-256"
        )
    if COMMIT_RE.fullmatch(str(baseline["android_commit"])) is None:
        raise AndroidRuntimeEvidenceError(
            "android_commit must be a full commit"
        )
    expected_baseline = _mapping(
        json.loads(BASELINE_PATH.read_text(encoding="utf-8")),
        "Phase C baseline",
    )
    expected_baseline_hash = hashlib.sha256(
        BASELINE_PATH.read_bytes()
    ).hexdigest()
    if baseline != {
        "baseline_id": expected_baseline["baseline_id"],
        "baseline_manifest_sha256": expected_baseline_hash,
        "android_commit": expected_baseline["source_snapshot"]["revision"],
    }:
        raise AndroidRuntimeEvidenceError(
            "runtime evidence is not bound to the exact Phase C baseline"
        )

    captures = package["captures"]
    if not isinstance(captures, list) or not captures:
        raise AndroidRuntimeEvidenceError(
            "captures must be a nonempty list"
        )
    gap_codes = set(contract["allowed_gap_codes"])
    summaries = [
        _validate_capture(value, index, gap_codes)
        for index, value in enumerate(captures)
    ]
    classes = sorted(
        {summary["evidence_class"] for summary in summaries}
    )
    return {
        "status": "valid",
        "contract_id": package["contract_id"],
        "contract_version": package["contract_version"],
        "privacy_classification": package["privacy_classification"],
        "baseline_manifest_sha256": baseline[
            "baseline_manifest_sha256"
        ],
        "capture_count": len(summaries),
        "evidence_classes": classes,
        "captures_by_class": {
            evidence_class: sum(
                summary["evidence_class"] == evidence_class
                for summary in summaries
            )
            for evidence_class in classes
        },
        "sample_total_by_class": {
            evidence_class: sum(
                summary["sample_total"]
                for summary in summaries
                if summary["evidence_class"] == evidence_class
            )
            for evidence_class in classes
        },
    }


def read_runtime_evidence_package(
    path: Path,
    *,
    allow_synthetic_fixture: bool = False,
) -> Mapping[str, Any]:
    """Read aggregate evidence only from RESULTS or the exact invented fixture."""

    resolved = path.expanduser().resolve()
    allowed = False
    try:
        resolved.relative_to(RESULTS_ROOT.resolve())
        allowed = True
    except ValueError:
        allowed = (
            allow_synthetic_fixture
            and resolved == SYNTHETIC_FIXTURE.resolve()
        )
    if not allowed:
        raise AndroidRuntimeEvidenceError(
            "runtime evidence input must remain under RESULTS; only the exact "
            "invented fixture is allowed with an explicit flag"
        )

    def reject_constant(_value: str) -> None:
        raise AndroidRuntimeEvidenceError(
            "runtime evidence contains non-finite JSON"
        )

    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AndroidRuntimeEvidenceError(
            "runtime evidence is unavailable or invalid JSON"
        ) from error
    return _mapping(value, "runtime evidence package")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
