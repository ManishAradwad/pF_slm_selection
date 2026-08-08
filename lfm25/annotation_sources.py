"""Frozen source loaders and workflow policies for the local annotation workbench."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .annotation_workbench import (
    ACTIVE_LEARNING_QUEUE_POLICY_VERSION,
    ACTIVE_LEARNING_QUEUE_TAGS,
    BLINDED_MODE,
    DECISIONS,
    PRIVATE_ROOT,
    QC_SAMPLE_DENOMINATOR,
    QC_SAMPLE_NUMERATOR,
    TRAINING_MODE,
    WORKBENCH_CONTRACT,
    WORKBENCH_MODES,
    WORKBENCH_SCHEMA_VERSION,
    WorkbenchError,
    WorkbenchSourceRow,
    WorkspaceDefinition,
    empty_annotation,
    exact_json_dumps,
    exact_json_loads,
    validate_annotation,
)
from .candidates import extract_protocol_candidates
from .private_data import (
    PrivateDataError,
    ensure_within,
    file_sha256,
    normalize_template,
    require_private_ignore,
)


MAX_SOURCE_LENGTH = 100_000
HANDBOOK_VERSION = "v1"
HANDBOOK_PATH = Path("docs/guides/ANNOTATION_HANDBOOK_V1.md")
_AMOUNT_SIGNAL_RE = re.compile(r"(?:\u20b9|\brs\.?|\binr\b)\s*[+]?\d", re.IGNORECASE)
_OTP_RE = re.compile(
    r"\b(?:otp|one[- ]time password|verification code|authenticate|do not share)\b",
    re.IGNORECASE,
)
_PENDING_RE = re.compile(
    r"\b(?:pending|failed|declined|blocked|unsuccessful|on hold|pre[- ]?authori[sz])\b",
    re.IGNORECASE,
)
_REQUEST_RE = re.compile(
    r"\b(?:request(?:ed)? to pay|collect request|payment due|amount due|bill due|"
    r"reminder|pay by)\b",
    re.IGNORECASE,
)
_REFUND_RE = re.compile(r"\b(?:refund(?:ed)?|reversal|reversed)\b", re.IGNORECASE)


def _read_exact_jsonl(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = exact_json_loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append((value, line.rstrip("\n")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkbenchError("a private workbench input could not be parsed") from exc
    return rows


def private_path(repo_root: Path, value: Path) -> Path:
    private_root = (repo_root / PRIVATE_ROOT).resolve()
    candidate = value if value.is_absolute() else repo_root / value
    try:
        return ensure_within(candidate, private_root)
    except PrivateDataError as exc:
        raise WorkbenchError("a workbench input is outside PRIVATE_DATA/lfm25") from exc


def _handbook_binding(repo_root: Path) -> dict[str, str]:
    try:
        handbook = ensure_within(repo_root / HANDBOOK_PATH, repo_root)
    except PrivateDataError as exc:
        raise WorkbenchError("the annotation handbook path is invalid") from exc
    if not handbook.is_file():
        raise WorkbenchError("the versioned annotation handbook is missing")
    return {
        "handbook_version": HANDBOOK_VERSION,
        "handbook_sha256": file_sha256(handbook),
    }


def _source_rows_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        visible = {
            "review_id": row["review_id"],
            "sender": row["sender"],
            "sms": row["sms"],
        }
        digest.update((exact_json_dumps(visible) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _character_to_utf8_offset(text: str, index: int) -> int:
    return len(text[:index].encode("utf-8"))


def _exact_text_span(sms: str, value: str, *, field: str) -> dict[str, Any]:
    start = sms.find(value)
    if start < 0:
        raise WorkbenchError(f"a completed legacy annotation has an ungrounded {field}")
    second = sms.find(value, start + 1)
    if second >= 0:
        raise WorkbenchError(f"a completed legacy annotation has an ambiguous {field}")
    return {
        "text": value,
        "start": _character_to_utf8_offset(sms, start),
        "end": _character_to_utf8_offset(sms, start + len(value)),
    }


def _legacy_amount_span(sms: str, amount: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(amount, bool) or not isinstance(amount, (int, Decimal)):
        raise WorkbenchError("a completed legacy annotation has an invalid amount")
    target = Decimal(amount)
    matches = [
        candidate
        for candidate in extract_protocol_candidates(sms).amounts
        if Decimal(str(candidate.value)) == target
    ]
    if len(matches) != 1:
        raise WorkbenchError("a completed legacy annotation has an ambiguous source amount")
    candidate = matches[0]
    if candidate.start is None or candidate.end is None or candidate.source_text is None:
        raise WorkbenchError("a completed legacy annotation has an ungrounded source amount")
    return str(candidate.value), {
        "text": candidate.source_text,
        "start": _character_to_utf8_offset(sms, candidate.start),
        "end": _character_to_utf8_offset(sms, candidate.end),
    }


def _legacy_annotation(row: Mapping[str, Any]) -> dict[str, Any] | None:
    decision = row.get("decision")
    if decision is None:
        return None
    notes = row.get("notes")
    if decision == "not_transaction":
        value = empty_annotation()
        value.update({"decision": decision, "notes": notes})
        return validate_annotation(value, str(row["sms"]), require_complete=True)
    if decision != "transaction":
        raise WorkbenchError("a completed legacy annotation has an invalid decision")
    sms = str(row["sms"])
    amount_decimal, amount_span = _legacy_amount_span(sms, row.get("amount"))
    account = row.get("account")
    counterparty = row.get("counterparty")
    if not isinstance(account, str) or not account.strip():
        raise WorkbenchError("a completed legacy annotation has an invalid account")
    value = empty_annotation()
    value.update(
        {
            "decision": decision,
            "amount_decimal": amount_decimal,
            "amount_span": amount_span,
            "type": row.get("type"),
            "account_span": _exact_text_span(sms, account, field="account"),
            "counterparty_span": (
                _exact_text_span(sms, counterparty, field="counterparty")
                if isinstance(counterparty, str) and counterparty
                else None
            ),
            "counterparty_absent": counterparty is None,
            "notes": notes,
        }
    )
    return validate_annotation(value, sms, require_complete=True)


def load_blinded_workspace(
    repo_root: Path,
    *,
    source_manifest: Path | None = None,
    review_file: Path | None = None,
    mapping_file: Path | None = None,
    metadata_file: Path | None = None,
    include_initial_annotations: bool = True,
) -> WorkspaceDefinition:
    """Load and bind the frozen reviewer package without exposing internals."""

    if not isinstance(include_initial_annotations, bool):
        raise WorkbenchError("the blinded annotation import option is invalid")
    from .blinded_review import (
        DEFAULT_MAPPING_FILE,
        DEFAULT_METADATA_FILE,
        DEFAULT_REVIEW_FILE,
        DEFAULT_SOURCE_MANIFEST,
        REVIEW_FIELDS,
        resolve_review_paths,
        run_validate,
    )

    repo = repo_root.resolve()
    supplied = {
        "source_manifest": source_manifest or DEFAULT_SOURCE_MANIFEST,
        "review_file": review_file or DEFAULT_REVIEW_FILE,
        "mapping_file": mapping_file or DEFAULT_MAPPING_FILE,
        "metadata_file": metadata_file or DEFAULT_METADATA_FILE,
    }
    report = run_validate(repo, **supplied)
    paths = resolve_review_paths(repo, **supplied)
    try:
        metadata_value = exact_json_loads(paths.metadata_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkbenchError("the blinded package metadata could not be parsed") from exc
    if not isinstance(metadata_value, dict):
        raise WorkbenchError("the blinded package metadata is invalid")
    review_rows = [row for row, _raw in _read_exact_jsonl(paths.review_file)]
    if len(review_rows) != report["test_rows"]:
        raise WorkbenchError("the blinded review row count changed during launch")

    prepared: list[WorkbenchSourceRow] = []
    for position, row in enumerate(review_rows):
        if tuple(row) != REVIEW_FIELDS:
            raise WorkbenchError("the blinded reviewer-facing schema changed during launch")
        initial = _legacy_annotation(row) if include_initial_annotations else None
        prepared.append(
            WorkbenchSourceRow(
                row_id=str(row["review_id"]),
                position=position,
                sender=str(row["sender"]),
                sms=str(row["sms"]),
                source_json=None,
                split=None,
                queue_tags=(),
                initial_annotation=initial,
                initial_reviewer=(str(row["reviewer"]) if initial is not None else None),
                initial_reviewed_at=(str(row["reviewed_at"]) if initial is not None else None),
            )
        )
    binding = {
        "contract": WORKBENCH_CONTRACT,
        "mode": BLINDED_MODE,
        **_handbook_binding(repo_root),
        "source_manifest_sha256": file_sha256(paths.source_manifest),
        "mapping_file_sha256": file_sha256(paths.mapping_file),
        "metadata_file_sha256": file_sha256(paths.metadata_file),
        "review_template_sha256": metadata_value.get("review_template_sha256"),
        "source_rows_sha256": _source_rows_digest(review_rows),
        "row_count": len(prepared),
        "ordering": "frozen_reviewer_file_order",
    }
    return WorkspaceDefinition(
        mode=BLINDED_MODE,
        rows=tuple(prepared),
        binding=binding,
        metadata={
            "schema_version": WORKBENCH_SCHEMA_VERSION,
            "package_version": metadata_value.get("package_version"),
            "test_rows": len(prepared),
            "completed_rows_at_bootstrap": report["completed_rows"],
            "raw_values_emitted_to_console": False,
        },
    )


def _source_sender_group(row: Mapping[str, Any]) -> str:
    private_hashes = row.get("private_hashes")
    if not isinstance(private_hashes, Mapping):
        raise WorkbenchError("a training-pool row is missing split identity")
    value = private_hashes.get("sender")
    if not isinstance(value, str) or not value:
        raise WorkbenchError("a training-pool row is missing split identity")
    return value


def _proposal_label_identity(proposal: Mapping[str, Any]) -> str | None:
    if proposal.get("schema_valid") is False or "label" not in proposal:
        return None
    try:
        return exact_json_dumps(proposal.get("label"))
    except (TypeError, ValueError):
        return None


def _queue_tags(
    row: Mapping[str, Any],
    *,
    sender_counts: Mapping[str, int],
    template_counts: Mapping[str, int],
) -> tuple[str, ...]:
    tags: set[str] = set()
    sms = str(row["sms"])
    raw_proposals = row.get("local_model_proposals")
    proposals = (
        [item for item in raw_proposals if isinstance(item, Mapping)]
        if isinstance(raw_proposals, list)
        else []
    )
    identities = {
        identity
        for proposal in proposals
        if (identity := _proposal_label_identity(proposal)) is not None
    }
    if len(identities) > 1:
        tags.add("model_disagreement")
    confidence_values = [
        Decimal(str(proposal["confidence"]))
        for proposal in proposals
        if isinstance(proposal.get("confidence"), (int, float, Decimal))
        and not isinstance(proposal.get("confidence"), bool)
    ]
    row_confidence = row.get("confidence")
    proposal_low = bool(confidence_values) and min(confidence_values) < Decimal("0.85")
    heuristic_low = (
        isinstance(row_confidence, (int, float, Decimal))
        and not isinstance(row_confidence, bool)
        and Decimal(str(row_confidence)) < Decimal("0.85")
    )
    if proposal_low or heuristic_low:
        tags.add("low_confidence_output")
    if row.get("candidate_oracle_covered") is False or row.get("candidate_coverage_miss") is True:
        tags.add("candidate_coverage_miss")
    if row.get("hard_negative_category") and _AMOUNT_SIGNAL_RE.search(sms):
        tags.add("hard_negative_with_amount")
    if _OTP_RE.search(sms):
        tags.add("otp_or_security")
    if _PENDING_RE.search(sms):
        tags.add("pending_failed_declined_or_hold")
    if _REQUEST_RE.search(sms):
        tags.add("payment_request_or_reminder")
    if _REFUND_RE.search(sms):
        tags.add("refund_or_reversal")
    candidates = extract_protocol_candidates(sms)
    source_counterparties = [item for item in candidates.counterparties if item.value is not None]
    if len(candidates.amounts) > 1 or len(candidates.accounts) > 1 or len(source_counterparties) > 1:
        tags.add("multiple_entities")
    sender_group = _source_sender_group(row)
    template_group = str(row["template_group"])
    if sender_counts[sender_group] <= 2 or template_counts[template_group] <= 2:
        tags.add("rare_sender_or_template")
    return tuple(name for name in ACTIVE_LEARNING_QUEUE_TAGS if name in tags)


def _validate_pool_row(row: Mapping[str, Any], seen: set[str]) -> None:
    record_hash = row.get("record_hash")
    if not isinstance(record_hash, str) or not record_hash or record_hash in seen:
        raise WorkbenchError("the training pool contains a missing or duplicate row ID")
    seen.add(record_hash)
    if row.get("split") not in {"train", "dev"}:
        raise WorkbenchError("the training pool contains an ineligible split")
    if not isinstance(row.get("sender"), str) or not isinstance(row.get("sms"), str):
        raise WorkbenchError("the training pool contains an invalid source row")
    if not row["sms"].strip() or len(row["sms"]) > MAX_SOURCE_LENGTH:
        raise WorkbenchError("the training pool contains an invalid source row")
    if not isinstance(row.get("template_group"), str) or not row["template_group"]:
        raise WorkbenchError("a training-pool row is missing split identity")
    _source_sender_group(row)


def load_training_workspace(
    repo_root: Path,
    *,
    pool_file: Path,
    sealed_manifest: Path = PRIVATE_ROOT / "split_manifest.jsonl",
) -> WorkspaceDefinition:
    """Load an explicit train/dev-only pool and fail closed on sealed-test overlap."""

    repo = repo_root.resolve()
    private_root = (repo / PRIVATE_ROOT).resolve()
    require_private_ignore(repo, private_root)
    pool = private_path(repo, pool_file)
    sealed = private_path(repo, sealed_manifest)
    if pool.resolve() == sealed.resolve():
        raise WorkbenchError("training curation requires an explicit train/dev-only pool")
    if not pool.is_file() or not sealed.is_file():
        raise WorkbenchError("a required private training input is missing")
    pool_pairs = _read_exact_jsonl(pool)
    sealed_pairs = _read_exact_jsonl(sealed)
    if not pool_pairs:
        raise WorkbenchError("the explicit training pool is empty")

    eligible_manifest_rows: dict[str, Mapping[str, Any]] = {}
    test_ids: set[str] = set()
    test_templates: set[str] = set()
    test_senders: set[str] = set()
    test_normalized: set[str] = set()
    sealed_ids: set[str] = set()
    for row, _raw in sealed_pairs:
        record_hash = row.get("record_hash")
        split = row.get("split")
        if (
            not isinstance(record_hash, str)
            or not record_hash
            or record_hash in sealed_ids
            or split not in {"train", "dev", "test"}
        ):
            raise WorkbenchError("the sealed split manifest has invalid row identity")
        sealed_ids.add(record_hash)
        if split in {"train", "dev"}:
            eligible_manifest_rows[record_hash] = row
            continue
        test_ids.add(record_hash)
        sender = row.get("sender")
        sms = row.get("sms")
        template_group = row.get("template_group")
        if (
            not isinstance(sender, str)
            or not sender.strip()
            or not isinstance(sms, str)
            or not sms.strip()
            or len(sms) > MAX_SOURCE_LENGTH
            or not isinstance(template_group, str)
            or not template_group.strip()
        ):
            raise WorkbenchError("the sealed test manifest has invalid split identity")
        test_templates.add(template_group)
        test_normalized.add(normalize_template(sms))
        try:
            test_senders.add(_source_sender_group(row))
        except WorkbenchError:
            raise WorkbenchError(
                "the sealed test manifest has invalid split identity"
            ) from None

    seen: set[str] = set()
    rows = [row for row, _raw in pool_pairs]
    for row in rows:
        _validate_pool_row(row, seen)
        record_hash = str(row["record_hash"])
        manifest_row = eligible_manifest_rows.get(record_hash)
        if manifest_row is None:
            raise WorkbenchError(
                "sealed test exclusion cannot be proven for a noncanonical training row"
            )
        if (
            row.get("split") != manifest_row.get("split")
            or row.get("sender") != manifest_row.get("sender")
            or row.get("sms") != manifest_row.get("sms")
            or row.get("template_group") != manifest_row.get("template_group")
            or _source_sender_group(row) != _source_sender_group(manifest_row)
        ):
            raise WorkbenchError(
                "sealed test exclusion cannot be proven for a changed training row"
            )
        if (
            str(row["record_hash"]) in test_ids
            or str(row["template_group"]) in test_templates
            or _source_sender_group(row) in test_senders
            or normalize_template(str(row["sms"])) in test_normalized
        ):
            raise WorkbenchError("the training pool overlaps the sealed test partition")

    split_templates: dict[str, set[str]] = {"train": set(), "dev": set()}
    split_senders: dict[str, set[str]] = {"train": set(), "dev": set()}
    split_normalized: dict[str, set[str]] = {"train": set(), "dev": set()}
    for row in rows:
        split = str(row["split"])
        split_templates[split].add(str(row["template_group"]))
        split_senders[split].add(_source_sender_group(row))
        split_normalized[split].add(normalize_template(str(row["sms"])))
    if (
        split_templates["train"] & split_templates["dev"]
        or split_senders["train"] & split_senders["dev"]
        or split_normalized["train"] & split_normalized["dev"]
    ):
        raise WorkbenchError("the training pool violates sender/template split isolation")

    sender_counts = Counter(_source_sender_group(row) for row in rows)
    template_counts = Counter(str(row["template_group"]) for row in rows)
    prepared: list[WorkbenchSourceRow] = []
    queue_counts: Counter[str] = Counter()
    for position, (row, raw) in enumerate(pool_pairs):
        tags = _queue_tags(row, sender_counts=sender_counts, template_counts=template_counts)
        queue_counts.update(tags)
        prepared.append(
            WorkbenchSourceRow(
                row_id=str(row["record_hash"]),
                position=position,
                sender=str(row["sender"]),
                sms=str(row["sms"]),
                source_json=raw,
                split=str(row["split"]),
                queue_tags=tags,
            )
        )

    binding = {
        "contract": WORKBENCH_CONTRACT,
        "mode": TRAINING_MODE,
        **_handbook_binding(repo_root),
        "pool_sha256": file_sha256(pool),
        "sealed_manifest_sha256": file_sha256(sealed),
        "row_count": len(prepared),
        "ordering": "explicit_pool_file_order",
        "active_learning_queue_policy_version": ACTIVE_LEARNING_QUEUE_POLICY_VERSION,
        "active_learning_queue_priority": list(ACTIVE_LEARNING_QUEUE_TAGS),
        "active_learning_queue_tie_break": "original_pool_position",
        "record_id_set_sha256": hashlib.sha256(
            ("\n".join(sorted(seen)) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return WorkspaceDefinition(
        mode=TRAINING_MODE,
        rows=tuple(prepared),
        binding=binding,
        metadata={
            "schema_version": WORKBENCH_SCHEMA_VERSION,
            "pool_rows": len(prepared),
            "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
            "queue_counts": dict(sorted(queue_counts.items())),
            "sealed_test_rows_eligible": 0,
            "raw_values_emitted_to_console": False,
        },
    )


def training_proposals(source_json: str, *, mode: str) -> dict[str, Any]:
    """Return training-only proposals after an explicit, separately audited reveal."""

    if mode != TRAINING_MODE:
        raise WorkbenchError("proposal reveal is unavailable in blinded-test mode")
    try:
        source = exact_json_loads(source_json)
    except (json.JSONDecodeError, ValueError) as exc:
        raise WorkbenchError("the private training source could not be parsed") from exc
    if not isinstance(source, Mapping):
        raise WorkbenchError("the private training source has an invalid schema")
    proposals: list[dict[str, Any]] = []
    raw_proposals = source.get("local_model_proposals")
    if isinstance(raw_proposals, list):
        for proposal in raw_proposals:
            if not isinstance(proposal, Mapping):
                continue
            proposals.append(
                {
                    key: proposal.get(key)
                    for key in (
                        "model_id",
                        "model_family",
                        "label",
                        "confidence",
                        "confidence_basis",
                        "schema_valid",
                    )
                    if key in proposal
                }
            )
    return {
        "revealed": True,
        "model_proposals": proposals,
        "heuristic_proposal": {
            "label": source.get("silver_label"),
            "confidence": source.get("confidence"),
            "reason_codes": source.get("heuristic_reason_codes", []),
        },
    }


def qc_requirements(
    rows: Sequence[Mapping[str, Any]],
    *,
    deterministic_seed: str,
) -> dict[str, tuple[str, ...]]:
    """Select the minimum single-reviewer QC queue deterministically."""

    requirements: dict[str, set[str]] = {}
    null_ids: list[str] = []
    for row in rows:
        row_id = row.get("row_id")
        annotation = row.get("annotation")
        if not isinstance(row_id, str) or not isinstance(annotation, Mapping):
            raise WorkbenchError("QC cannot start until every source row is completed")
        if annotation.get("uncertain") is True or annotation.get("decision") not in DECISIONS:
            raise WorkbenchError("QC cannot start while an annotation is unresolved")
        reasons = requirements.setdefault(row_id, set())
        if annotation.get("decision") == "transaction":
            reasons.add("transaction_second_pass")
        else:
            null_ids.append(row_id)
        if annotation.get("notes"):
            reasons.add("noted_second_pass")
        if row.get("ever_uncertain") is True:
            reasons.add("uncertain_second_pass")

    sample_size = (
        len(null_ids) * QC_SAMPLE_NUMERATOR + QC_SAMPLE_DENOMINATOR - 1
    ) // QC_SAMPLE_DENOMINATOR
    ranked_nulls = sorted(
        null_ids,
        key=lambda row_id: hashlib.sha256(
            f"{deterministic_seed}\0{row_id}".encode("utf-8")
        ).hexdigest(),
    )
    for row_id in ranked_nulls[:sample_size]:
        requirements[row_id].add("deterministic_null_sample_10pct")
    return {
        row_id: tuple(sorted(reasons))
        for row_id, reasons in requirements.items()
        if reasons
    }


def public_row(
    source: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    mode: str,
    total_rows: int,
) -> dict[str, Any]:
    """Project a storage row through an explicit workflow-specific allowlist."""

    if mode not in WORKBENCH_MODES:
        raise WorkbenchError("the workbench mode is invalid")
    value = {
        "review_id": source["row_id"],
        "position": source["position"],
        "total_rows": total_rows,
        "sender": source["sender"],
        "sms": source["sms"],
        "status": state.get("status", "pending"),
        "revision": state.get("revision", 0),
        "annotation": state.get("annotation") or empty_annotation(),
        "qc_required": bool(state.get("qc_required", False)),
        "qc_status": state.get("qc_status"),
    }
    if mode == TRAINING_MODE:
        value["queue_tags"] = list(source.get("queue_tags", ()))
    return value
