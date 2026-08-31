"""Build the complete ignored private corpus as one canonical source of truth."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..analyzer import DeterministicSmsAnalyzer
from ..currency import CurrencyContext
from ..labels import EventState, OperationalClass, WeakConfidence, WeakFacets
from ..provenance import (
    PrivateArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    ensure_private_directory,
    file_sha256,
    load_or_create_secret,
    object_sha256,
    require_private_output,
)
from ..triage import evaluate_triage
from ..types import CandidateKind, Disposition
from .grouping import Grouping, build_grouping, template_hash
from .pools import PoolInput, assign_pools, leakage_audit
from .reports import aggregate_reports


CORPUS_CONTRACT = "pocketfinancer.corpus-record/1"


def build_private_corpus(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = _resolve_within(repo_root, config_path)
    config = _read_object(config_path, "corpus run configuration")
    _validate_config(config)
    source_path = _resolve_within(repo_root, Path(config["source_path"]))
    output_root = require_private_output(repo_root, repo_root / config["output_root"])
    ensure_private_directory(output_root)
    key_path = require_private_output(repo_root, repo_root / config["source_id_key_path"])
    source_key = load_or_create_secret(key_path)
    source_key_id = hashlib.sha256(b"sms-source-key\0" + source_key).hexdigest()[:16]

    source_hash = file_sha256(source_path)
    config_hash = file_sha256(config_path)
    implementation_hash = _implementation_hash()
    run_id = hashlib.sha256(
        f"{source_hash}:{config_hash}:{implementation_hash}:{source_key_id}".encode()
    ).hexdigest()[:20]
    runs_root = output_root / "runs"
    ensure_private_directory(runs_root)
    final_run = runs_root / run_id
    if final_run.exists():
        summary = _read_object(final_run / "reports" / "summary.json", "existing corpus summary")
        atomic_write_json(output_root / "CURRENT.json", {"run_id": run_id})
        return summary

    raw_rows = _read_source_rows(source_path)
    if len(raw_rows) != int(config["expected_source_rows"]):
        raise PrivateArtifactError("source row count does not match the configured archive count")
    profiles = tuple(config["profile_ids"])
    analyzer = DeterministicSmsAnalyzer(CurrencyContext(config["primary_currency"], profiles))

    working: list[dict[str, Any]] = []
    pool_inputs: list[PoolInput] = []
    raw_ids: set[str] = set()
    source_ids: set[str] = set()
    for index, row in enumerate(raw_rows):
        source_record_id = str(row.get("id", ""))
        raw_identity = f"{source_record_id}\0{index}"
        source_id = "src_" + hmac.new(
            source_key, f"source-id\0{raw_identity}".encode(), hashlib.sha256
        ).hexdigest()[:32]
        if source_id in source_ids:
            raise PrivateArtifactError("canonical source ID collision detected")
        source_ids.add(source_id)
        raw_ids.add(source_record_id)
        body = row.get("text")
        sender = row.get("sender")
        timestamp = row.get("date")
        is_outgoing = row.get("is_from_me")
        input_valid = isinstance(body, str) and bool(body.strip())
        body_value = body if isinstance(body, str) else ""
        sender_value = sender if isinstance(sender, str) else ""
        timestamp_value = timestamp if isinstance(timestamp, str) else ""
        analysis = analyzer.analyze(
            body_value,
            operation_id=source_id,
            is_outgoing=is_outgoing if isinstance(is_outgoing, bool) else None,
            input_valid=input_valid,
        )
        triage = evaluate_triage(analysis)
        grouping = build_grouping(source_key, body_value, sender_value, timestamp_value)
        weak = _weak_facets(analysis, triage)
        item = {
            "source_id": source_id,
            "source_record_id": source_record_id,
            "source_row_index": index,
            "body": body_value,
            "sender": sender_value,
            "timestamp": timestamp_value,
            "service": row.get("service") if isinstance(row.get("service"), str) else None,
            "is_outgoing": is_outgoing if isinstance(is_outgoing, bool) else None,
            "analysis": analysis.to_dict(),
            "triage": triage,
            "weak": weak,
            "grouping": grouping,
        }
        working.append(item)
        pool_inputs.append(
            PoolInput(
                source_id=source_id,
                timestamp=timestamp_value,
                exact_body_hash=grouping.exact_body_hash,
                normalized_template_hash=grouping.normalized_template_hash,
                sender_template_group_hash=grouping.sender_template_group_hash,
            )
        )
    if len(source_ids) != len(raw_rows):
        raise PrivateArtifactError("canonical source IDs are not one-to-one with source rows")

    regression_templates = _reference_template_hashes(
        repo_root / config["regression_fixture_path"], source_key, reviewed_only=False
    )
    legacy_path = repo_root / config["legacy_review_path"]
    legacy_templates = _reference_template_hashes(
        legacy_path, source_key, reviewed_only=True
    )
    assignments = assign_pools(
        pool_inputs,
        regression_template_hashes=regression_templates,
        legacy_template_hashes=legacy_templates,
        later_time_cutoff=config["later_time_cutoff"],
    )
    leakage = leakage_audit(pool_inputs, assignments)
    if not leakage["passed"]:
        raise PrivateArtifactError("protected pool leakage audit failed")

    records = [_canonical_record(item, assignments[item["source_id"]], run_id) for item in working]
    summary = aggregate_reports(records)
    if summary["row_count"] != int(config["expected_source_rows"]):
        raise PrivateArtifactError("canonical manifest completeness gate failed")
    legacy_asset, legacy_report = _legacy_review_asset(legacy_path, working, source_key)
    provenance = {
        "contract": "pocketfinancer.corpus-provenance/1",
        "run_id": run_id,
        "source_sha256": source_hash,
        "configuration_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "source_id_key_id": source_key_id,
        "analysis_config_hash": analyzer.currency_context.config_hash,
        "row_count": len(records),
        "raw_source_id_unique_count": len(raw_ids),
        "canonical_source_id_unique_count": len(source_ids),
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.staging-", dir=runs_root))
    os.chmod(staging, 0o700)
    _write_run(
        staging,
        records,
        leakage,
        summary,
        provenance,
        legacy_asset,
        legacy_report,
    )
    os.replace(staging, final_run)
    atomic_write_json(output_root / "CURRENT.json", {"run_id": run_id})
    return summary


def _canonical_record(item: dict[str, Any], pool: str, run_id: str) -> dict[str, Any]:
    triage = item["triage"]
    weak: WeakFacets = item["weak"]
    grouping: Grouping = item["grouping"]
    return {
        "contract": CORPUS_CONTRACT,
        "source_id": item["source_id"],
        "source": {"body": item["body"], "sender": item["sender"]},
        "source_metadata": {
            "source_record_id": item["source_record_id"],
            "source_row_index": item["source_row_index"],
            "timestamp": item["timestamp"],
            "service": item["service"],
            "is_outgoing": item["is_outgoing"],
        },
        "analysis": item["analysis"],
        "weak_facets": {
            "disposition": triage.disposition.value,
            "selector_action": triage.selector_action.value,
            "operational_class": weak.operational_class.value,
            "event_state": weak.event_state.value,
            "financial_family": weak.financial_family,
            "payment_rail": weak.payment_rail,
            "confidence": weak.confidence.value,
            "reason_codes": list(weak.reason_codes),
        },
        "grouping": asdict(grouping),
        "pool": pool,
        "review_state": "unreviewed",
        "provenance": {"corpus_run_id": run_id},
    }


def _weak_facets(analysis, triage) -> WeakFacets:
    cue_kinds = {cue.kind for cue in analysis.cues}
    rails = sorted(
        cue.reason_code.removeprefix("rail_")
        for cue in analysis.cues
        if cue.kind == "payment_rail"
    )
    rail = rails[0] if len(set(rails)) == 1 else ("unknown" if rails else None)
    account_types = {
        candidate.value.get("account_type")
        for candidate in analysis.candidates_of(CandidateKind.ACCOUNT)
        if not candidate.explicit_absence
    }
    if rail == "upi":
        family = "upi_transfer"
    elif "card" in account_types:
        family = "card_purchase"
        rail = rail or "card"
    elif analysis.candidates_of(CandidateKind.DIRECTION):
        family = "bank_transfer"
    else:
        family = None

    if "reliable_outgoing_metadata" in triage.reason_codes or "invalid_input" in triage.reason_codes:
        operational = OperationalClass.INVALID_OUTGOING
        state = EventState.NO_EVENT
        confidence = WeakConfidence.HIGH
    elif triage.disposition == Disposition.INVOKE:
        operational = OperationalClass.POSTED_CANDIDATE
        state = EventState.POSTED
        confidence = WeakConfidence.MEDIUM
    elif triage.disposition == Disposition.RETAIN_REVIEW:
        operational = OperationalClass.AMBIGUOUS
        state = EventState.UNKNOWN
        confidence = WeakConfidence.LOW
    elif cue_kinds & {"failure", "negation", "pending", "due", "request"}:
        operational = OperationalClass.FINANCIAL_NON_POSTED
        state = EventState.NOT_POSTED
        confidence = WeakConfidence.HIGH
    else:
        operational = OperationalClass.NON_FINANCIAL
        state = EventState.NO_EVENT
        confidence = WeakConfidence.HIGH
    return WeakFacets(
        operational_class=operational,
        event_state=state,
        financial_family=family,
        payment_rail=rail,
        confidence=confidence,
        reason_codes=triage.reason_codes,
    )


def _write_run(
    root: Path,
    records: list[dict[str, Any]],
    leakage: dict[str, Any],
    summary: dict[str, Any],
    provenance: dict[str, Any],
    legacy_asset: list[dict[str, Any]],
    legacy_report: dict[str, Any],
) -> None:
    reports = root / "reports"
    queues = root / "annotation_queues"
    ensure_private_directory(reports)
    ensure_private_directory(queues)
    atomic_write_jsonl(root / "canonical_manifest.jsonl", records)
    atomic_write_jsonl(
        root / "deterministic_analysis.jsonl",
        ({"source_id": row["source_id"], "analysis": row["analysis"]} for row in records),
    )
    atomic_write_jsonl(
        root / "weak_operational_segregation.jsonl",
        (
            {"source_id": row["source_id"], "weak_facets": row["weak_facets"]}
            for row in records
        ),
    )
    atomic_write_jsonl(
        root / "grouping.jsonl",
        ({"source_id": row["source_id"], **row["grouping"]} for row in records),
    )
    atomic_write_jsonl(
        root / "pool_assignments.jsonl",
        ({"source_id": row["source_id"], "pool": row["pool"]} for row in records),
    )
    for pool in sorted({row["pool"] for row in records}):
        atomic_write_jsonl(
            queues / f"{pool}.jsonl",
            (
                {
                    "source_id": row["source_id"],
                    "queue": pool,
                    "blind_first": pool in {"protected_test", "later_time_holdout"},
                }
                for row in records
                if row["pool"] == pool
            ),
        )
    atomic_write_jsonl(
        queues / "retain_review.jsonl",
        (
            {
                "source_id": row["source_id"],
                "queue": "retain_review",
                "reason_codes": row["weak_facets"]["reason_codes"],
            }
            for row in records
            if row["weak_facets"]["disposition"] == "retain_review"
        ),
    )
    hard_negative_reasons = {
        "credential_otp",
        "non_posted_failure",
        "request_or_authorization",
        "amount_due",
        "balance_information",
        "promotion",
    }
    atomic_write_jsonl(
        queues / "hard_negative_challenge.jsonl",
        (
            {"source_id": row["source_id"], "queue": "hard_negative_challenge"}
            for row in records
            if hard_negative_reasons & set(row["weak_facets"]["reason_codes"])
        ),
    )
    atomic_write_jsonl(root / "legacy_review_asset.jsonl", legacy_asset)
    atomic_write_json(reports / "summary.json", summary)
    atomic_write_json(reports / "candidate_coverage.json", summary["candidate_coverage"])
    atomic_write_json(
        reports / "prefilter.json",
        {
            "dispositions": summary["prefilter_dispositions"],
            "selector_actions": summary["selector_actions"],
        },
    )
    atomic_write_json(reports / "leakage_audit.json", leakage)
    atomic_write_json(reports / "legacy_review_migration.json", legacy_report)
    atomic_write_json(root / "provenance.json", provenance)


def _legacy_review_asset(
    path: Path, working: list[dict[str, Any]], key: bytes
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return [], {"total": 0, "unique_match": 0, "ambiguous_match": 0, "missing": 0}
    indexes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in working:
        grouping: Grouping = item["grouping"]
        indexes[(grouping.exact_body_hash, grouping.sender_hash)].append(item["source_id"])
    rows = _read_jsonl(path)
    asset: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("decision") not in {"transaction", "not_transaction"}:
            continue
        body = row.get("sms") if isinstance(row.get("sms"), str) else ""
        sender = row.get("sender") if isinstance(row.get("sender"), str) else ""
        grouping = build_grouping(key, body, sender, "")
        matches = indexes.get((grouping.exact_body_hash, grouping.sender_hash), [])
        status = "unique_match" if len(matches) == 1 else "ambiguous_match" if matches else "missing"
        counts[status] += 1
        asset.append(
            {
                "migration_status": status,
                "source_ids": matches,
                "legacy_annotation": row,
                "canonical_conversion_status": "not_attempted_requires_validated_conversion",
            }
        )
    return asset, {"total": len(asset), **{name: counts[name] for name in ("unique_match", "ambiguous_match", "missing")}}


def _reference_template_hashes(
    path: Path, key: bytes, *, reviewed_only: bool
) -> set[str]:
    if not path.is_file():
        return set()
    rows = _read_jsonl(path)
    values = set()
    for row in rows:
        if reviewed_only and row.get("decision") not in {"transaction", "not_transaction"}:
            continue
        body = row.get("sms") if isinstance(row.get("sms"), str) else row.get("text")
        if isinstance(body, str):
            values.add(template_hash(key, body))
    return values


def _read_source_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateArtifactError("private source archive could not be read") from exc
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise PrivateArtifactError("private source archive must be a JSON row array")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateArtifactError("private reference asset could not be read") from exc
    if not all(isinstance(row, dict) for row in values):
        raise PrivateArtifactError("private reference asset contains a non-object row")
    return values


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateArtifactError(f"{description} could not be read") from exc
    if not isinstance(value, dict):
        raise PrivateArtifactError(f"{description} must be an object")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "source_path",
        "expected_source_rows",
        "output_root",
        "primary_currency",
        "profile_ids",
        "later_time_cutoff",
        "source_id_key_path",
        "regression_fixture_path",
        "legacy_review_path",
    }
    if not required <= set(config):
        raise PrivateArtifactError("corpus configuration is missing required fields")
    if config.get("offline_retention") != "retain_every_source_row":
        raise PrivateArtifactError("corpus configuration must retain every source row")
    if config.get("build_sft_targets") is not False:
        raise PrivateArtifactError("corpus rebuild must not create SFT targets")


def _resolve_within(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    if resolved != root and root not in resolved.parents:
        raise PrivateArtifactError("configured path is outside the repository")
    return resolved


def _implementation_hash() -> str:
    paths = [
        Path(__file__),
        Path(__file__).with_name("grouping.py"),
        Path(__file__).with_name("pools.py"),
        Path(__file__).with_name("reports.py"),
        Path(__file__).parents[1] / "analyzer.py",
        Path(__file__).parents[1] / "triage.py",
    ]
    return object_sha256({str(path.name): file_sha256(path) for path in paths})
