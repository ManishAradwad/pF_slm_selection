"""Synthetic native-trace import and encrypted-store safety tests."""

from __future__ import annotations

import base64
import hashlib
import json
import types
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest

from pocketfinancer_sms import cli
from pocketfinancer_sms.analyzer import DeterministicSmsAnalyzer
from pocketfinancer_sms.corpus.grouping import build_grouping
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.provenance import PrivateArtifactError
from pocketfinancer_sms.triage import evaluate_triage
from pocketfinancer_sms.types import EvidenceSpan
from pocketfinancer_sms.workbench.native_import import NativeTraceImporter
from pocketfinancer_sms.workbench.secure_store import SecureWorkbenchStore
from pocketfinancer_sms.workbench.service import WorkbenchService
from pocketfinancer_sms.workbench.store import WorkbenchStore


RELEASE_HASH = "7a5ebf1ecdb56374a0d6563a28c3b0856134ea5c14c6a552a423b615f1b9e265"


class SyntheticDecryptor:
    def __init__(self, plaintext: bytes) -> None:
        self.plaintext = plaintext
        self.calls = 0

    def decrypt(self, *, ciphertext: bytes, nonce: bytes, key_protection: str) -> bytes:
        assert ciphertext
        assert len(nonce) == 12
        assert key_protection == "os_protected"
        self.calls += 1
        return self.plaintext


def _store(tmp_path: Path, pool: str) -> tuple[WorkbenchStore, str]:
    source_id = "src_" + "c" * 32
    body = "INR 25 was debited from account **1234 at SYNTH SHOP."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india"))
    ).analyze(body, operation_id=source_id, is_outgoing=False)
    triage = evaluate_triage(analysis)
    timestamp = "2024-03-01T00:00:00Z"
    grouping = build_grouping(b"native-import-key" * 2, body, "SYNTH", timestamp)
    record = {
        "contract": "pocketfinancer.corpus-record/1",
        "source_id": source_id,
        "source": {"body": body, "sender": "SYNTH"},
        "source_metadata": {
            "source_record_id": "synthetic-native",
            "source_row_index": 0,
            "timestamp": timestamp,
            "service": "SMS",
            "is_outgoing": False,
        },
        "analysis": analysis.to_dict(),
        "weak_facets": {
            "disposition": triage.disposition.value,
            "selector_action": triage.selector_action.value,
            "operational_class": "posted_candidate",
            "event_state": "posted",
            "financial_family": "merchant_payment",
            "payment_rail": "unknown",
            "confidence": "medium",
            "reason_codes": list(triage.reason_codes),
        },
        "grouping": asdict(grouping),
        "pool": pool,
        "review_state": "unreviewed",
        "provenance": {"corpus_run_id": "synthetic-native-run"},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    store = WorkbenchStore(tmp_path / "private" / "workbench.sqlite3")
    store.import_manifest(manifest, corpus_run_id="synthetic-native-run")
    return store, source_id


def _bundle(tmp_path: Path, source_id: str) -> tuple[Path, SyntheticDecryptor]:
    source_ref = hashlib.sha256(source_id.encode()).hexdigest()
    operation_id = str(uuid.uuid4())
    record = {
        "operation_id": operation_id,
        "source_ref_hash": source_ref,
        "operation": {"state": "retain_review"},
        "configuration": {"contract": "pocketfinancer.processing-config/1"},
        "deterministic_analysis": {
            "candidates": [
                {"candidate_id": "amt_synthetic"},
                {"candidate_id": "dir_synthetic"},
            ]
        },
        "machine_proposal": {
            "decision": "posted",
            "amount": "amt_synthetic",
            "direction": "dir_synthetic",
        },
        "reconstructed_result": {"amount_minor_units": 9_223_372_036_854_775_807},
        "persistence_decision": {"result": "blocked_by_mode"},
        "processing_trace": [],
        "native_feedback": [
            {
                "corrections": [
                    {
                        "classification": "supplied_source_supported_candidate_miss",
                        "field": "counterparty",
                    }
                ]
            },
            {
                "contract": "pocketfinancer.user-feedback/2",
                "field_corrections": [
                    {
                        "field": "amount",
                        "classification": "selected_existing_candidate",
                        "evidence": asdict(EvidenceSpan.from_source(
                            "INR 25 was debited from account **1234 at SYNTH SHOP.", 0, 6
                        )),
                    },
                    {
                        "field": "account",
                        "classification": "selected_existing_candidate",
                        "evidence": {
                            **asdict(EvidenceSpan.from_source(
                                "INR 25 was debited from account **1234 at SYNTH SHOP.", 32, 38
                            )),
                            "text": "WRONG",
                        },
                    },
                ],
            },
        ],
    }
    plaintext = json.dumps(
        {"contract": "pocketfinancer.native-trace-payload/1", "records": [record]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    ciphertext = b"synthetic-authenticated-ciphertext"
    envelope = {
        "contract": "pocketfinancer.native-trace-bundle/1",
        "transfer_id": str(uuid.uuid4()),
        "created_at_epoch_ms": 1_700_000_000_000,
        "source_platform": "ios",
        "release_id": "native-integration-v1",
        "release_manifest_sha256": RELEASE_HASH,
        "explicit_consent": True,
        "purpose": "local_workbench_inspection",
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_protection": "os_protected",
            "nonce_base64": base64.b64encode(b"123456789012").decode(),
        },
        "source_ref_hashes": [source_ref],
        "record_count": 1,
        "payload_ciphertext_base64": base64.b64encode(ciphertext).decode(),
        "payload_ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
    }
    path = tmp_path / "native-trace.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path, SyntheticDecryptor(plaintext)


def test_native_import_keeps_provenance_and_candidate_misses_separate(tmp_path: Path) -> None:
    store, source_id = _store(tmp_path, "annotation_training")
    bundle, decryptor = _bundle(tmp_path, source_id)
    importer = NativeTraceImporter(store, expected_release_hash=RELEASE_HASH)

    receipt = importer.import_bundle(bundle, decryptor=decryptor, imported_at_epoch_ms=1)
    replay = importer.import_bundle(bundle, decryptor=decryptor, imported_at_epoch_ms=2)
    row = WorkbenchService(store, native_trace_importer=importer).view_row(
        source_id, "synthetic-reviewer"
    )

    assert receipt["record_count"] == 1 and receipt["replayed"] is False
    assert replay["replayed"] is True
    assert row["native_traces"][0]["provenance_class"] == "native_trace_import"
    assert row["native_traces"][0]["candidate_coverage"] == {
        "deterministic_candidate_count": 2,
        "selected_candidate_count": 2,
        "candidate_miss_count": 1,
    }
    assert '"amount_minor_units":9223372036854775807' in row["native_traces"][0][
        "record_json"
    ]
    assert row["latest_annotation"] is None
    assert len(row["native_suggestions"]) == 1
    assert row["native_suggestions"][0]["field"] == "amount"
    assert row["native_suggestions"][0]["evidence"]["text"] == "INR 25"
    service = WorkbenchService(store, native_trace_importer=importer)
    assert service.list_rows(
        reviewer_id="synthetic-reviewer", filters={"imported_feedback": "available"},
    )["total"] == 1
    assert service.list_rows(
        reviewer_id="synthetic-reviewer", filters={"imported_feedback": "correction"},
    )["total"] == 1


def test_blind_pool_match_is_rejected_before_decryption(tmp_path: Path) -> None:
    store, source_id = _store(tmp_path, "protected_test")
    bundle, decryptor = _bundle(tmp_path, source_id)
    importer = NativeTraceImporter(store, expected_release_hash=RELEASE_HASH)

    with pytest.raises(PrivateArtifactError, match="blind protected pool"):
        importer.import_bundle(bundle, decryptor=decryptor, imported_at_epoch_ms=1)
    assert decryptor.calls == 0


def test_secure_store_fails_closed_without_sqlcipher(monkeypatch, tmp_path: Path) -> None:
    import pocketfinancer_sms.workbench.secure_store as secure_store

    real_import = secure_store.importlib.import_module

    def reject_cipher(name: str):
        if name in {"sqlcipher3", "pysqlcipher3.dbapi2"}:
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(secure_store.importlib, "import_module", reject_cipher)
    with pytest.raises(PrivateArtifactError, match="requires a SQLCipher"):
        SecureWorkbenchStore(
            tmp_path / "encrypted.sqlite3",
            key_provider=type("Key", (), {"get_or_create_key": lambda *_: b"x" * 32})(),
        )


def test_native_import_cli_emits_only_aggregate_receipt(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    store, source_id = _store(tmp_path, "annotation_training")
    bundle, decryptor = _bundle(tmp_path, source_id)
    monkeypatch.setattr(cli, "_current_workbench", lambda _root: (store, tmp_path))
    monkeypatch.setattr(cli, "_release_manifest_sha256", lambda _root: RELEASE_HASH)
    monkeypatch.setattr(
        cli,
        "KeyringKeyProvider",
        lambda: type("Keyring", (), {"get_existing_key": lambda *_: b"x" * 32})(),
    )
    monkeypatch.setattr(cli, "AesGcmBundleDecryptor", lambda _key: decryptor)
    monkeypatch.setattr(cli, "SecureWorkbenchStore", WorkbenchStore)

    result = cli.main(
        ["--repo-root", str(tmp_path), "import-native-trace", "--bundle", str(bundle)]
    )

    assert result == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {"record_count": 1, "replayed": False, "status": "imported"}
    assert str(bundle) not in json.dumps(receipt)


def test_cli_refuses_legacy_plaintext_workbench(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repository"
    legacy = (
        repo_root
        / "PRIVATE_DATA"
        / "sms_processing"
        / "workbench"
        / "workbench.sqlite3"
    )
    WorkbenchStore(legacy)
    private_root = repo_root / "PRIVATE_DATA" / "sms_processing"
    monkeypatch.setattr(cli, "_private_root", lambda _root: private_root)
    monkeypatch.setattr(cli, "require_private_output", lambda _root, path: path.resolve())

    with pytest.raises(PrivateArtifactError, match="migrate-secure-workbench"):
        cli._current_workbench(repo_root)


def test_secure_export_requires_consent_and_writes_only_ciphertext(
    monkeypatch, tmp_path: Path
) -> None:
    store, source_id = _store(tmp_path, "annotation_training")
    revision = store.append_annotation_revision(
        source_id=source_id, reviewer_id="synthetic-reviewer",
        expected_revision=0, status="submitted",
        payload={"decision": "none"},
        canonical_label={"contract": "synthetic-test-label", "decision": "none"},
        created_at_epoch_ms=1,
    )
    selection = [{
        "source_id": source_id, "reviewer_id": "synthetic-reviewer",
        "revision": revision["revision"], "revision_hash": revision["revision_hash"],
    }]
    secure = object.__new__(SecureWorkbenchStore)
    secure.database_path = store.database_path
    secure._key_provider = type(
        "Key", (), {"get_or_create_key": lambda *_: b"x" * 32}
    )()
    secure._key_id = "synthetic-key"
    secure.connect = store.connect

    class SyntheticAesGcm:
        def __init__(self, key: bytes) -> None:
            assert key == b"x" * 32

        def encrypt(self, nonce: bytes, plaintext: bytes, associated_data: bytes) -> bytes:
            assert len(nonce) == 12
            assert plaintext
            assert associated_data == b"pocketfinancer.workbench-export/2"
            return hashlib.sha256(plaintext).digest()

    import pocketfinancer_sms.workbench.secure_store as secure_store

    real_import = secure_store.importlib.import_module
    monkeypatch.setattr(
        secure_store.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(AESGCM=SyntheticAesGcm)
        if name == "cryptography.hazmat.primitives.ciphers.aead"
        else real_import(name),
    )

    with pytest.raises(PrivateArtifactError, match="explicit consent"):
        secure.export_labels(tmp_path / "exports")
    receipt = secure.export_labels(
        tmp_path / "exports", explicit_consent=True,
        selected_revisions=selection,
    )
    envelope_path = tmp_path / "exports" / f"workbench-export-{receipt['export_id']}.json"
    envelope_text = envelope_path.read_text(encoding="utf-8")

    assert receipt["encrypted"] is True
    assert '"payload_ciphertext_base64"' in envelope_text
    assert "SYNTH SHOP" not in envelope_text
    assert "synthetic-native-run" not in json.dumps(receipt)
