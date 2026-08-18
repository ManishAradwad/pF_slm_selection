"""Privacy-safe local workbench foundations for PocketFinancer Semantic V2.

This module deliberately has no network, model, or mobile-application integration.
Private packages may only be read from or written beneath ``PRIVATE_DATA`` and
generated aggregate reports may only be written beneath ``RESULTS``.  Pure mapping
APIs make invented unit fixtures possible without weakening those filesystem gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from typing import Any, Mapping, Sequence

from .semantic_v2 import SemanticRecord, project_initial_auto_post, validate_semantic_v2


WORKBENCH_CONTRACT_ID = "pocketfinancer_workbench_v2"
WORKBENCH_CONTRACT_VERSION = 2
ANONYMIZED_CONTRACT_ID = "pocketfinancer_workbench_v2_anonymized"
ANONYMIZED_CONTRACT_VERSION = 1

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKBENCH_CONTRACT_PATH = (
    _REPOSITORY_ROOT / "configs/contracts/pocketfinancer-workbench-v2.json"
)
PRIVATE_DATA_ROOT = _REPOSITORY_ROOT / "PRIVATE_DATA"
RESULTS_ROOT = _REPOSITORY_ROOT / "RESULTS"
SYNTHETIC_FIXTURE_PATH = (
    _REPOSITORY_ROOT / "tests/fixtures/pocketfinancer_workbench_v2_synthetic.json"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_ID_RE = re.compile(r"[a-z][a-z0-9_-]{2,63}\Z")
_PRIVATE_GROUP_RE = re.compile(r"grp_[0-9a-f]{16,64}\Z")
_SYNTHETIC_GROUP_RE = re.compile(r"invented_[a-z0-9_-]{3,48}\Z")
_RFC3339_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


class WorkbenchV2Error(ValueError):
    """A package violates the frozen local Workbench V2 contract."""


@dataclass(frozen=True)
class ResolvedAnnotationRow:
    """One locally resolved row; callers must never log or publish its raw fields."""

    row_id: str
    message: str
    split: str
    groups: Mapping[str, str]
    safety_tags: tuple[str, ...]
    gold: SemanticRecord
    gold_mapping: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise WorkbenchV2Error("package must contain only finite JSON values") from error


def package_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical package digest used by prediction provenance."""

    return _sha256_bytes(_canonical_json_bytes(value))


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkbenchV2Error(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise WorkbenchV2Error(f"{path} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise WorkbenchV2Error(f"{path} has invalid keys")


def _text(value: Any, path: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise WorkbenchV2Error(f"{path} must be non-empty text")
    if pattern is not None and not pattern.fullmatch(value):
        raise WorkbenchV2Error(f"{path} has an invalid format")
    return value


def _nullable_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _enum_text(value: Any, allowed: Sequence[str], path: str) -> str:
    text = _text(value, path)
    if text not in allowed:
        raise WorkbenchV2Error(f"{path} has an unsupported value")
    return text


def workbench_contract() -> dict[str, Any]:
    """Load and verify the frozen Workbench V2 declaration and Semantic pins."""

    try:
        value = json.loads(WORKBENCH_CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkbenchV2Error("Workbench V2 contract is unavailable or invalid") from error
    contract = _object(value, "workbench_contract")
    if (
        contract.get("contract_id") != WORKBENCH_CONTRACT_ID
        or contract.get("contract_version") != WORKBENCH_CONTRACT_VERSION
        or contract.get("status") != "frozen"
    ):
        raise WorkbenchV2Error("Workbench V2 contract identity is invalid")

    semantic = _object(contract.get("semantic_contract"), "semantic_contract")
    for path_key, hash_key in (
        ("schema_path", "schema_sha256"),
        ("reference_path", "reference_sha256"),
        ("conformance_path", "conformance_sha256"),
    ):
        relative_path = _text(semantic.get(path_key), f"semantic_contract.{path_key}")
        expected = _text(
            semantic.get(hash_key), f"semantic_contract.{hash_key}", pattern=_SHA256_RE
        )
        if _sha256_file(_REPOSITORY_ROOT / relative_path) != expected:
            raise WorkbenchV2Error(f"frozen Semantic V2 artifact drifted: {path_key}")
    return dict(contract)


def _semantic_binding() -> dict[str, Any]:
    semantic = workbench_contract()["semantic_contract"]
    return {
        "id": semantic["id"],
        "version": semantic["version"],
        "schema_sha256": semantic["schema_sha256"],
        "reference_sha256": semantic["reference_sha256"],
        "conformance_sha256": semantic["conformance_sha256"],
    }


def create_annotation_package(
    *,
    package_id: str,
    created_at: str,
    annotation_policy_id: str,
    annotation_policy_version: int,
    source_revision: str,
    source_digest_sha256: str,
    privacy_classification: str = "private_local_only",
    construction_attestation: str = "User-authorized local material; never publication-approved.",
) -> dict[str, Any]:
    """Create an empty, version-bound package without reading any source data."""

    _text(package_id, "package_id", pattern=_OPAQUE_ID_RE)
    _text(created_at, "created_at", pattern=_RFC3339_UTC_RE)
    _text(annotation_policy_id, "annotation_policy_id", pattern=_OPAQUE_ID_RE)
    if isinstance(annotation_policy_version, bool) or not isinstance(annotation_policy_version, int):
        raise WorkbenchV2Error("annotation_policy_version must be an integer")
    if annotation_policy_version < 1:
        raise WorkbenchV2Error("annotation_policy_version must be positive")
    _text(source_revision, "source_revision")
    _text(source_digest_sha256, "source_digest_sha256", pattern=_SHA256_RE)
    contract = workbench_contract()
    privacy = contract["privacy"]
    classification = _enum_text(
        privacy_classification,
        [privacy["private_classification"], privacy["synthetic_classification"]],
        "privacy_classification",
    )
    synthetic = classification == privacy["synthetic_classification"]
    return {
        "workbench_contract": {
            "id": WORKBENCH_CONTRACT_ID,
            "version": WORKBENCH_CONTRACT_VERSION,
        },
        "semantic_contract": _semantic_binding(),
        "privacy": {
            "classification": classification,
            "contains_message_text": True,
            "contains_private_data": not synthetic,
            "intended_storage": (
                "versioned_test_fixture_only" if synthetic else "PRIVATE_DATA_only"
            ),
            "construction_attestation": _text(
                construction_attestation, "construction_attestation"
            ),
        },
        "provenance": {
            "package_id": package_id,
            "created_at": created_at,
            "annotation_policy_id": annotation_policy_id,
            "annotation_policy_version": annotation_policy_version,
            "source_revision": source_revision,
            "source_digest_sha256": source_digest_sha256,
            "parent_package_sha256": None,
        },
        "split_policy": dict(contract["split_policy"]),
        "adjudication_policy": dict(contract["adjudication_policy"]),
        "rows": [],
    }


def _validate_semantic_binding(value: Any) -> None:
    binding = _object(value, "semantic_contract")
    expected = _semantic_binding()
    _exact_keys(binding, set(expected), "semantic_contract")
    if dict(binding) != expected:
        raise WorkbenchV2Error("package Semantic V2 binding does not match frozen artifacts")


def _validate_privacy(value: Any, contract: Mapping[str, Any]) -> str:
    privacy = _object(value, "privacy")
    _exact_keys(
        privacy,
        {
            "classification",
            "contains_message_text",
            "contains_private_data",
            "intended_storage",
            "construction_attestation",
        },
        "privacy",
    )
    declaration = contract["privacy"]
    classification = _enum_text(
        privacy["classification"],
        [declaration["private_classification"], declaration["synthetic_classification"]],
        "privacy.classification",
    )
    if privacy["contains_message_text"] is not True:
        raise WorkbenchV2Error("annotation packages must declare contained message text")
    synthetic = classification == declaration["synthetic_classification"]
    if privacy["contains_private_data"] is not (not synthetic):
        raise WorkbenchV2Error("privacy.contains_private_data contradicts classification")
    expected_storage = "versioned_test_fixture_only" if synthetic else "PRIVATE_DATA_only"
    if privacy["intended_storage"] != expected_storage:
        raise WorkbenchV2Error("privacy.intended_storage contradicts classification")
    _text(privacy["construction_attestation"], "privacy.construction_attestation")
    return classification


def _validate_provenance(value: Any) -> None:
    provenance = _object(value, "provenance")
    _exact_keys(
        provenance,
        {
            "package_id",
            "created_at",
            "annotation_policy_id",
            "annotation_policy_version",
            "source_revision",
            "source_digest_sha256",
            "parent_package_sha256",
        },
        "provenance",
    )
    _text(provenance["package_id"], "provenance.package_id", pattern=_OPAQUE_ID_RE)
    _text(provenance["created_at"], "provenance.created_at", pattern=_RFC3339_UTC_RE)
    _text(
        provenance["annotation_policy_id"],
        "provenance.annotation_policy_id",
        pattern=_OPAQUE_ID_RE,
    )
    version = provenance["annotation_policy_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise WorkbenchV2Error("provenance.annotation_policy_version must be positive")
    _text(provenance["source_revision"], "provenance.source_revision")
    _text(
        provenance["source_digest_sha256"],
        "provenance.source_digest_sha256",
        pattern=_SHA256_RE,
    )
    parent = provenance["parent_package_sha256"]
    if parent is not None:
        _text(parent, "provenance.parent_package_sha256", pattern=_SHA256_RE)


def _validate_fixed_policy(value: Any, expected: Mapping[str, Any], path: str) -> None:
    mapping = _object(value, path)
    if dict(mapping) != dict(expected):
        raise WorkbenchV2Error(f"{path} does not match the frozen Workbench V2 contract")


def _group_id(value: Any, path: str, *, synthetic: bool) -> str:
    pattern = _SYNTHETIC_GROUP_RE if synthetic else _PRIVATE_GROUP_RE
    return _text(value, path, pattern=pattern)


def _resolved_mapping(
    annotations: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
    *,
    message: str,
    row_path: str,
) -> Mapping[str, Any] | None:
    status = adjudication["status"]
    selected = adjudication["selected_annotation_id"]
    supplied = adjudication["semantic_record"]
    adjudicator = adjudication["adjudicator_id"]
    reason = adjudication["reason_code"]
    if status == "pending":
        if any(item is not None for item in (selected, supplied, adjudicator)):
            raise WorkbenchV2Error(f"{row_path}.adjudication pending fields must be null")
        _nullable_text(reason, f"{row_path}.adjudication.reason_code")
        return None
    if status == "excluded":
        if any(item is not None for item in (selected, supplied, adjudicator)):
            raise WorkbenchV2Error(f"{row_path}.adjudication excluded fields must be null")
        _text(reason, f"{row_path}.adjudication.reason_code", pattern=_OPAQUE_ID_RE)
        return None

    _text(adjudicator, f"{row_path}.adjudication.adjudicator_id", pattern=_OPAQUE_ID_RE)
    annotator_ids = {annotation["annotator_id"] for annotation in annotations}
    if adjudicator in annotator_ids:
        raise WorkbenchV2Error(f"{row_path}.adjudicator must be independent")
    if len(annotations) < 2:
        raise WorkbenchV2Error(f"{row_path} needs two independent annotations")
    if (selected is None) == (supplied is None):
        raise WorkbenchV2Error(
            f"{row_path}.adjudication must select one annotation or supply one record"
        )
    _text(reason, f"{row_path}.adjudication.reason_code", pattern=_OPAQUE_ID_RE)
    if selected is not None:
        _text(selected, f"{row_path}.adjudication.selected_annotation_id", pattern=_OPAQUE_ID_RE)
        matches = [item["semantic_record"] for item in annotations if item["annotation_id"] == selected]
        if len(matches) != 1:
            raise WorkbenchV2Error(f"{row_path}.adjudication selects an unknown annotation")
        return matches[0]
    mapping = _object(supplied, f"{row_path}.adjudication.semantic_record")
    validate_semantic_v2(mapping, message=message)
    return mapping


def _validate_row(
    value: Any,
    *,
    index: int,
    classification: str,
    contract: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, str], ResolvedAnnotationRow | None]:
    path = f"rows[{index}]"
    row = _object(value, path)
    _exact_keys(
        row,
        {"row_id", "message", "split", "groups", "safety_tags", "annotations", "adjudication"},
        path,
    )
    row_id = _text(row["row_id"], f"{path}.row_id", pattern=_OPAQUE_ID_RE)
    message = row["message"]
    if not isinstance(message, str):
        raise WorkbenchV2Error(f"{path}.message must be text")
    split_policy = contract["split_policy"]
    split = _enum_text(row["split"], split_policy["allowed_splits"], f"{path}.split")
    synthetic = classification == contract["privacy"]["synthetic_classification"]
    groups = _object(row["groups"], f"{path}.groups")
    expected_group_keys = set(split_policy["group_keys"])
    _exact_keys(groups, expected_group_keys, f"{path}.groups")
    normalized_groups = {
        key: _group_id(groups[key], f"{path}.groups.{key}", synthetic=synthetic)
        for key in split_policy["group_keys"]
    }

    tags = _array(row["safety_tags"], f"{path}.safety_tags")
    if not tags:
        raise WorkbenchV2Error(f"{path}.safety_tags must not be empty")
    allowed_tags = contract["safety_tags"]
    normalized_tags = tuple(
        _enum_text(item, allowed_tags, f"{path}.safety_tags") for item in tags
    )
    if len(set(normalized_tags)) != len(normalized_tags):
        raise WorkbenchV2Error(f"{path}.safety_tags contains duplicates")
    if "ordinary" in normalized_tags and len(normalized_tags) != 1:
        raise WorkbenchV2Error(f"{path}.ordinary cannot be combined with safety tags")

    annotations_value = _array(row["annotations"], f"{path}.annotations")
    annotations: list[Mapping[str, Any]] = []
    annotation_ids: set[str] = set()
    annotator_ids: set[str] = set()
    for annotation_index, item in enumerate(annotations_value):
        annotation_path = f"{path}.annotations[{annotation_index}]"
        annotation = _object(item, annotation_path)
        _exact_keys(annotation, {"annotation_id", "annotator_id", "semantic_record"}, annotation_path)
        annotation_id = _text(
            annotation["annotation_id"], f"{annotation_path}.annotation_id", pattern=_OPAQUE_ID_RE
        )
        annotator_id = _text(
            annotation["annotator_id"], f"{annotation_path}.annotator_id", pattern=_OPAQUE_ID_RE
        )
        if annotation_id in annotation_ids or annotator_id in annotator_ids:
            raise WorkbenchV2Error(f"{annotation_path} is not independent and unique")
        annotation_ids.add(annotation_id)
        annotator_ids.add(annotator_id)
        record = _object(annotation["semantic_record"], f"{annotation_path}.semantic_record")
        validate_semantic_v2(record, message=message)
        annotations.append(annotation)

    adjudication = _object(row["adjudication"], f"{path}.adjudication")
    _exact_keys(
        adjudication,
        {
            "status",
            "adjudicator_id",
            "selected_annotation_id",
            "semantic_record",
            "reason_code",
        },
        f"{path}.adjudication",
    )
    adjudication_status = _enum_text(
        adjudication["status"],
        contract["adjudication_policy"]["allowed_statuses"],
        f"{path}.adjudication.status",
    )
    normalized_adjudication = {**adjudication, "status": adjudication_status}
    resolved = _resolved_mapping(
        annotations,
        normalized_adjudication,
        message=message,
        row_path=path,
    )
    if resolved is None:
        return row_id, split, normalized_groups, None
    record = validate_semantic_v2(resolved, message=message)
    return (
        row_id,
        split,
        normalized_groups,
        ResolvedAnnotationRow(
            row_id=row_id,
            message=message,
            split=split,
            groups=normalized_groups,
            safety_tags=normalized_tags,
            gold=record,
            gold_mapping=resolved,
        ),
    )


def validate_annotation_package(value: Any) -> dict[str, Any]:
    """Validate a package and return only aggregate-safe structural counts."""

    package = _object(value, "package")
    _exact_keys(
        package,
        {
            "workbench_contract",
            "semantic_contract",
            "privacy",
            "provenance",
            "split_policy",
            "adjudication_policy",
            "rows",
        },
        "package",
    )
    identity = _object(package["workbench_contract"], "workbench_contract")
    _exact_keys(identity, {"id", "version"}, "workbench_contract")
    if identity != {"id": WORKBENCH_CONTRACT_ID, "version": WORKBENCH_CONTRACT_VERSION}:
        raise WorkbenchV2Error("package has an unsupported Workbench V2 identity")
    contract = workbench_contract()
    _validate_semantic_binding(package["semantic_contract"])
    classification = _validate_privacy(package["privacy"], contract)
    _validate_provenance(package["provenance"])
    _validate_fixed_policy(package["split_policy"], contract["split_policy"], "split_policy")
    _validate_fixed_policy(
        package["adjudication_policy"],
        contract["adjudication_policy"],
        "adjudication_policy",
    )

    rows = _array(package["rows"], "rows")
    row_ids: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    counts = {"rows": 0, "pending": 0, "resolved": 0, "excluded": 0}
    split_counts = {split: 0 for split in contract["split_policy"]["allowed_splits"]}
    for index, item in enumerate(rows):
        row_id, split, groups, resolved = _validate_row(
            item,
            index=index,
            classification=classification,
            contract=contract,
        )
        if row_id in row_ids:
            raise WorkbenchV2Error(f"rows[{index}].row_id is duplicated")
        row_ids.add(row_id)
        for group_key, group_id in groups.items():
            prior = group_splits.setdefault((group_key, group_id), split)
            if prior != split:
                raise WorkbenchV2Error("sender/template group crosses split boundary")
        status = item["adjudication"]["status"]
        counts[status] += 1
        counts["rows"] += 1
        split_counts[split] += 1
        if resolved is not None and status != "resolved":
            raise WorkbenchV2Error("only resolved rows may produce gold records")
    return {
        "contract_id": WORKBENCH_CONTRACT_ID,
        "contract_version": WORKBENCH_CONTRACT_VERSION,
        "privacy_classification": classification,
        "counts": counts,
        "split_counts": split_counts,
        "package_sha256": package_sha256(package),
    }


def resolved_annotation_rows(value: Any, *, split: str | None = None) -> tuple[ResolvedAnnotationRow, ...]:
    """Return validated resolved rows for local scoring without serializing them."""

    validate_annotation_package(value)
    package = _object(value, "package")
    contract = workbench_contract()
    classification = package["privacy"]["classification"]
    if split is not None:
        _enum_text(split, contract["split_policy"]["allowed_splits"], "split")
    resolved_rows: list[ResolvedAnnotationRow] = []
    for index, item in enumerate(package["rows"]):
        _, row_split, _, resolved = _validate_row(
            item,
            index=index,
            classification=classification,
            contract=contract,
        )
        if resolved is not None and (split is None or row_split == split):
            resolved_rows.append(resolved)
    return tuple(resolved_rows)


def _semantic_shape(record: SemanticRecord) -> dict[str, Any]:
    projection = project_initial_auto_post(record)
    return {
        "scope": record.scope.value,
        "posting_status": record.posting_status.value,
        "event_cardinality": record.event_cardinality.value,
        "events": [
            {
                "has_amount": event.amount is not None,
                "currency": event.amount.currency if event.amount is not None else None,
                "has_exact_minor_units": (
                    event.amount is not None and event.amount.minor_units is not None
                ),
                "has_direction": event.direction is not None,
                "has_account": event.account is not None,
                "counterparty_state": event.counterparty.state.value,
            }
            for event in record.events
        ],
        "automatic_post_eligible": projection.eligible,
        "automatic_post_ineligibility_reasons": [reason.value for reason in projection.reasons],
    }


def anonymize_annotation_package(
    value: Any,
    *,
    secret: bytes,
    export_nonce: str | None = None,
) -> dict[str, Any]:
    """Remove raw fields and pseudonymize row IDs for a local row-level export.

    The returned artifact is safer for local analysis but remains row-level and is
    explicitly not publication-approved.  The caller-held secret is never returned.
    """

    summary = validate_annotation_package(value)
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise WorkbenchV2Error("anonymization secret must contain at least 32 bytes")
    nonce = export_nonce if export_nonce is not None else secrets.token_hex(16)
    _text(nonce, "export_nonce", pattern=re.compile(r"[a-zA-Z0-9_-]{16,128}\Z"))
    package = _object(value, "package")
    package_id = package["provenance"]["package_id"]
    rows = []
    for row in resolved_annotation_rows(package):
        scoped_id = f"{nonce}\0{package_id}\0{row.row_id}".encode()
        anonymous_id = hmac.new(secret, scoped_id, hashlib.sha256).hexdigest()
        rows.append(
            {
                "anonymous_row_id": anonymous_id,
                "split": row.split,
                "safety_tags": list(row.safety_tags),
                "semantic_shape": _semantic_shape(row.gold),
            }
        )
    return {
        "anonymized_contract": {
            "id": ANONYMIZED_CONTRACT_ID,
            "version": ANONYMIZED_CONTRACT_VERSION,
        },
        "semantic_contract": dict(package["semantic_contract"]),
        "source_package_sha256": summary["package_sha256"],
        "privacy": {
            "classification": workbench_contract()["privacy"]["anonymized_classification"],
            "contains_message_text": False,
            "contains_source_identifiers": False,
            "contains_exact_financial_values": False,
            "publication_approved": False,
        },
        "export_nonce": nonce,
        "rows": rows,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: None)
    except (OSError, json.JSONDecodeError) as error:
        raise WorkbenchV2Error("local package is unavailable or invalid JSON") from error


def read_annotation_package(path: Path, *, allow_synthetic_fixture: bool = False) -> dict[str, Any]:
    """Read only an ignored private package or the one committed invented fixture."""

    resolved = path.expanduser().resolve()
    synthetic_allowed = allow_synthetic_fixture and resolved == SYNTHETIC_FIXTURE_PATH.resolve()
    if not _is_within(resolved, PRIVATE_DATA_ROOT) and not synthetic_allowed:
        raise WorkbenchV2Error("annotation input must remain under PRIVATE_DATA")
    value = _object(_read_json(resolved), "package")
    validate_annotation_package(value)
    return dict(value)


def write_private_artifact(path: Path, value: Mapping[str, Any]) -> None:
    """Write a raw or anonymized row-level artifact only beneath PRIVATE_DATA."""

    resolved = path.expanduser().resolve()
    if not _is_within(resolved, PRIVATE_DATA_ROOT):
        raise WorkbenchV2Error("row-level output must remain under PRIVATE_DATA")
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved.write_bytes(_canonical_json_bytes(value))
    resolved.chmod(0o600)


def write_aggregate_report(path: Path, value: Mapping[str, Any]) -> None:
    """Write an aggregate-only report beneath RESULTS after a privacy scan."""

    forbidden_keys = {
        "message",
        "row_id",
        "anonymous_row_id",
        "semantic_record",
        "prediction",
        "gold",
        "source_mapping",
    }

    def scan(item: Any) -> None:
        if isinstance(item, Mapping):
            if forbidden_keys & set(item):
                raise WorkbenchV2Error("aggregate report contains a forbidden row-level field")
            for child in item.values():
                scan(child)
        elif isinstance(item, list):
            for child in item:
                scan(child)

    scan(value)
    resolved = path.expanduser().resolve()
    if not _is_within(resolved, RESULTS_ROOT):
        raise WorkbenchV2Error("aggregate output must remain under RESULTS")
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved.write_bytes(_canonical_json_bytes(value))
    resolved.chmod(0o600)
