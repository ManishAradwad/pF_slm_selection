"""Encrypted native trace import with provenance and blind-pool separation."""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Protocol

from ..provenance import PrivateArtifactError, object_sha256
from .store import WorkbenchStore


PROTECTED_POOLS = frozenset({"protected_test", "later_time_holdout"})
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 10_000


class NativeBundleDecryptor(Protocol):
    def decrypt(self, *, ciphertext: bytes, nonce: bytes, key_protection: str) -> bytes: ...


class AesGcmBundleDecryptor:
    """Optional AES-256-GCM decryptor; key acquisition stays outside this class."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise PrivateArtifactError("native trace import key must contain exactly 32 bytes")
        self._key = key

    def decrypt(self, *, ciphertext: bytes, nonce: bytes, key_protection: str) -> bytes:
        if key_protection not in {"os_protected", "user_passphrase"}:
            raise PrivateArtifactError("native trace key protection is unsupported")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise PrivateArtifactError(
                "AES-GCM native trace import requires the optional cryptography package"
            ) from exc
        try:
            return AESGCM(self._key).decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise PrivateArtifactError("native trace bundle authentication failed") from exc


class NativeTraceImporter:
    def __init__(self, store: WorkbenchStore, *, expected_release_hash: str) -> None:
        if not _is_sha256(expected_release_hash):
            raise ValueError("expected release hash must be SHA-256")
        self.store = store
        self.expected_release_hash = expected_release_hash
        self._initialize_tables()

    def import_bundle(
        self,
        bundle_path: Path,
        *,
        decryptor: NativeBundleDecryptor,
        imported_at_epoch_ms: int,
    ) -> dict[str, Any]:
        if not bundle_path.is_file() or bundle_path.stat().st_size > MAX_BUNDLE_BYTES:
            raise PrivateArtifactError("native trace bundle is missing or exceeds the size limit")
        try:
            envelope = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateArtifactError("native trace bundle is not valid JSON") from exc
        _validate_envelope(envelope, self.expected_release_hash)
        self._reject_blind_pool_matches(envelope["source_ref_hashes"])

        ciphertext = _decode_base64(envelope["payload_ciphertext_base64"], "ciphertext")
        if hashlib.sha256(ciphertext).hexdigest() != envelope["payload_ciphertext_sha256"]:
            raise PrivateArtifactError("native trace ciphertext hash does not match")
        nonce = _decode_base64(envelope["encryption"]["nonce_base64"], "nonce")
        if len(nonce) != 12:
            raise PrivateArtifactError("native trace AES-GCM nonce must contain 12 bytes")
        plaintext = decryptor.decrypt(
            ciphertext=ciphertext,
            nonce=nonce,
            key_protection=envelope["encryption"]["key_protection"],
        )
        if len(plaintext) > MAX_BUNDLE_BYTES:
            raise PrivateArtifactError("native trace plaintext exceeds the size limit")
        try:
            payload = json.loads(plaintext)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateArtifactError("native trace plaintext is not valid JSON") from exc
        records = _validate_payload(payload, envelope)
        manifest_hash = object_sha256(envelope)

        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT manifest_hash, record_count FROM native_trace_imports WHERE transfer_id = ?",
                (envelope["transfer_id"],),
            ).fetchone()
            if existing is not None:
                if existing["manifest_hash"] != manifest_hash:
                    raise PrivateArtifactError("native trace transfer id was reused")
                return {
                    "transfer_id": envelope["transfer_id"],
                    "record_count": int(existing["record_count"]),
                    "replayed": True,
                }
            for record in records:
                coverage = _candidate_coverage(record)
                connection.execute(
                    """
                    INSERT INTO native_trace_records(
                        transfer_id, operation_id, source_ref_hash, source_platform,
                        record_json, deterministic_candidate_count,
                        selected_candidate_count, candidate_miss_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        envelope["transfer_id"],
                        record["operation_id"],
                        record["source_ref_hash"],
                        envelope["source_platform"],
                        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                        coverage["deterministic_candidate_count"],
                        coverage["selected_candidate_count"],
                        coverage["candidate_miss_count"],
                    ),
                )
            connection.execute(
                """
                INSERT INTO native_trace_imports(
                    transfer_id, manifest_hash, source_platform, release_id,
                    release_manifest_sha256, imported_at_epoch_ms, record_count,
                    provenance_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'native_trace_import')
                """,
                (
                    envelope["transfer_id"],
                    manifest_hash,
                    envelope["source_platform"],
                    envelope["release_id"],
                    envelope["release_manifest_sha256"],
                    imported_at_epoch_ms,
                    len(records),
                ),
            )
        return {
            "transfer_id": envelope["transfer_id"],
            "record_count": len(records),
            "replayed": False,
        }

    def records_for_source(self, source_id: str) -> list[dict[str, Any]]:
        source_ref_hash = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        with self.store.connect() as connection:
            rows = connection.execute(
                """
                SELECT source_platform, record_json, deterministic_candidate_count,
                       selected_candidate_count, candidate_miss_count
                FROM native_trace_records WHERE source_ref_hash = ?
                ORDER BY transfer_id, operation_id
                """,
                (source_ref_hash,),
            ).fetchall()
        return [
            {
                "provenance_class": "native_trace_import",
                "source_platform": row["source_platform"],
                # Keep the canonical JSON as text at the browser boundary. JavaScript
                # numbers cannot represent every signed 64-bit money value exactly.
                "record_json": row["record_json"],
                "candidate_coverage": {
                    "deterministic_candidate_count": row["deterministic_candidate_count"],
                    "selected_candidate_count": row["selected_candidate_count"],
                    "candidate_miss_count": row["candidate_miss_count"],
                },
            }
            for row in rows
        ]

    def _reject_blind_pool_matches(self, source_ref_hashes: list[str]) -> None:
        if not source_ref_hashes:
            return
        with self.store.connect() as connection:
            protected = connection.execute(
                "SELECT source_id FROM corpus_rows WHERE pool IN ('protected_test', 'later_time_holdout')"
            ).fetchall()
        protected_hashes = {
            hashlib.sha256(row["source_id"].encode("utf-8")).hexdigest() for row in protected
        }
        if protected_hashes.intersection(source_ref_hashes):
            raise PrivateArtifactError("native trace bundle matches a blind protected pool")

    def _initialize_tables(self) -> None:
        with self.store.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS native_trace_imports(
                    transfer_id TEXT PRIMARY KEY,
                    manifest_hash TEXT NOT NULL,
                    source_platform TEXT NOT NULL,
                    release_id TEXT NOT NULL,
                    release_manifest_sha256 TEXT NOT NULL,
                    imported_at_epoch_ms INTEGER NOT NULL,
                    record_count INTEGER NOT NULL,
                    provenance_class TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS native_trace_records(
                    transfer_id TEXT NOT NULL REFERENCES native_trace_imports(transfer_id)
                        DEFERRABLE INITIALLY DEFERRED,
                    operation_id TEXT NOT NULL,
                    source_ref_hash TEXT NOT NULL,
                    source_platform TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    deterministic_candidate_count INTEGER NOT NULL,
                    selected_candidate_count INTEGER NOT NULL,
                    candidate_miss_count INTEGER NOT NULL,
                    PRIMARY KEY(transfer_id, operation_id)
                );
                CREATE INDEX IF NOT EXISTS native_trace_source_idx
                    ON native_trace_records(source_ref_hash, source_platform);
                """
            )


def _validate_envelope(value: Any, expected_release_hash: str) -> None:
    if not isinstance(value, dict):
        raise PrivateArtifactError("native trace bundle must be an object")
    required = {
        "contract", "transfer_id", "created_at_epoch_ms", "source_platform", "release_id",
        "release_manifest_sha256", "explicit_consent", "purpose", "encryption",
        "source_ref_hashes", "record_count", "payload_ciphertext_base64",
        "payload_ciphertext_sha256",
    }
    if set(value) != required:
        raise PrivateArtifactError("native trace bundle fields do not match contract /1")
    try:
        transfer_id = str(uuid.UUID(value["transfer_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise PrivateArtifactError("native trace transfer id is invalid") from exc
    if transfer_id != value["transfer_id"]:
        raise PrivateArtifactError("native trace transfer id must be canonical lowercase")
    if (
        value["contract"] != "pocketfinancer.native-trace-bundle/1"
        or value["source_platform"] not in {"android", "ios"}
        or value["explicit_consent"] is not True
        or value["purpose"] != "local_workbench_inspection"
        or value["release_manifest_sha256"] != expected_release_hash
        or not isinstance(value["record_count"], int)
        or isinstance(value["record_count"], bool)
        or not 0 <= value["record_count"] <= MAX_RECORDS
    ):
        raise PrivateArtifactError("native trace bundle policy validation failed")
    encryption = value["encryption"]
    if not isinstance(encryption, dict) or set(encryption) != {
        "algorithm", "key_protection", "nonce_base64"
    }:
        raise PrivateArtifactError("native trace encryption metadata is invalid")
    if encryption["algorithm"] != "AES-256-GCM":
        raise PrivateArtifactError("native trace encryption algorithm is unsupported")
    refs = value["source_ref_hashes"]
    if not isinstance(refs, list) or len(refs) != len(set(refs)) or not all(
        _is_sha256(item) for item in refs
    ):
        raise PrivateArtifactError("native trace source references are invalid")


def _validate_payload(payload: Any, envelope: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"contract", "records"}:
        raise PrivateArtifactError("native trace payload shape is invalid")
    if payload["contract"] != "pocketfinancer.native-trace-payload/1":
        raise PrivateArtifactError("native trace payload contract is unsupported")
    records = payload["records"]
    if not isinstance(records, list) or len(records) != envelope["record_count"]:
        raise PrivateArtifactError("native trace payload record count does not match")
    operation_ids: set[str] = set()
    references: set[str] = set()
    required = {
        "operation_id", "source_ref_hash", "operation", "configuration",
        "deterministic_analysis", "machine_proposal", "reconstructed_result",
        "persistence_decision", "processing_trace", "native_feedback",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != required:
            raise PrivateArtifactError("native trace record shape is invalid")
        try:
            operation_id = str(uuid.UUID(record["operation_id"]))
        except (ValueError, TypeError, AttributeError) as exc:
            raise PrivateArtifactError("native trace operation id is invalid") from exc
        if operation_id != record["operation_id"] or operation_id in operation_ids:
            raise PrivateArtifactError("native trace operation ids are invalid or duplicated")
        if not _is_sha256(record["source_ref_hash"]):
            raise PrivateArtifactError("native trace source reference is invalid")
        operation_ids.add(operation_id)
        references.add(record["source_ref_hash"])
        if not isinstance(record["processing_trace"], list) or not isinstance(
            record["native_feedback"], list
        ):
            raise PrivateArtifactError("native trace history fields must be arrays")
    if references != set(envelope["source_ref_hashes"]):
        raise PrivateArtifactError("native trace source references do not match the envelope")
    return records


def _candidate_coverage(record: dict[str, Any]) -> dict[str, int]:
    analysis = record["deterministic_analysis"]
    candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
    candidate_ids = {
        item.get("candidate_id") for item in candidates if isinstance(item, dict)
    } - {None}
    proposal = record["machine_proposal"]
    selected = set()
    if isinstance(proposal, dict) and proposal.get("decision") == "posted":
        selected = {
            proposal.get(name) for name in ("amount", "direction", "account", "counterparty")
        } - {None}
    feedback = record["native_feedback"]
    misses = sum(
        1
        for event in feedback
        if isinstance(event, dict)
        for correction in event.get("corrections", [])
        if isinstance(correction, dict)
        and correction.get("classification") == "supplied_source_supported_candidate_miss"
    )
    return {
        "deterministic_candidate_count": len(candidate_ids),
        "selected_candidate_count": len(selected.intersection(candidate_ids)),
        "candidate_miss_count": misses,
    }


def _decode_base64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise PrivateArtifactError(f"native trace {label} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise PrivateArtifactError(f"native trace {label} is not valid base64") from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
