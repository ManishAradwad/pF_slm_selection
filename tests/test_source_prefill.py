from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lfm25.annotation_service import AnnotationService
from lfm25.annotation_sources import source_prefill_for_sms
from lfm25.annotation_store import WorkbenchBindingError, WorkbenchStore
from lfm25.annotation_workbench import (
    ASSISTED_METHODOLOGY,
    HUMAN_VERIFIED_METHODOLOGY,
    SOURCE_PREFILL_OFF,
    SOURCE_PREFILL_POLICY_VERSION,
    SOURCE_PREFILL_UNAMBIGUOUS,
    TRAINING_MODE,
    WorkspaceDefinition,
    WorkbenchError,
    WorkbenchSourceRow,
    empty_annotation,
    exact_json_dumps,
    exact_json_loads,
)
from lfm25.candidates import extract_unambiguous_source_fields


ROW_ID = "invented-source-prefill-transaction"
SENDER = "SYNTH-DEMO-BANK"
SMS = (
    "[SYNTHETIC DEMO ONLY - NO PRIVATE DATA] Acme Demo Bank: "
    "INR 42.50 was debited from A/c XX1234 at Paper Kite Cafe."
)
SAFE_PREFILL_KEYS = frozenset(
    {
        "policy_version",
        "amount_decimal",
        "amount_span",
        "type",
        "account_span",
        "counterparty_span",
    }
)
FORBIDDEN_PREFILL_KEYS = frozenset({"decision", "counterparty_absent"})
PREFILL_HASHES = {
    "source_prefill_rows_digest_sha256": "1" * 64,
    "source_prefill_android_profile_sha256": "2" * 64,
    "source_prefill_candidate_contract_sha256": "3" * 64,
    "source_prefill_android_contract_sha256": "4" * 64,
    "source_prefill_candidate_extractor_sha256": "5" * 64,
    "source_prefill_annotation_policy_sha256": "6" * 64,
}


def _span(sms: str, text: str) -> dict[str, object]:
    character_start = sms.index(text)
    character_end = character_start + len(text)
    return {
        "text": text,
        "start": len(sms[:character_start].encode("utf-8")),
        "end": len(sms[:character_end].encode("utf-8")),
    }


def _prefill() -> dict[str, Any]:
    return {
        "policy_version": SOURCE_PREFILL_POLICY_VERSION,
        "amount_decimal": "42.50",
        "amount_span": _span(SMS, "INR 42.50"),
        "type": "debit",
        "account_span": _span(SMS, "A/c XX1234"),
        "counterparty_span": _span(SMS, "Paper Kite Cafe"),
    }


def _annotation() -> dict[str, Any]:
    value = empty_annotation()
    value.update(
        {
            "decision": "transaction",
            "amount_decimal": "42.50",
            "amount_span": _span(SMS, "INR 42.50"),
            "type": "debit",
            "account_span": _span(SMS, "A/c XX1234"),
            "counterparty_span": _span(SMS, "Paper Kite Cafe"),
            "counterparty_absent": False,
            "notes": "Invented reviewer verification only.",
        }
    )
    return value


def _binding(
    *,
    source_prefill: str = SOURCE_PREFILL_UNAMBIGUOUS,
    methodology: str = ASSISTED_METHODOLOGY,
) -> dict[str, Any]:
    row_ids = f"{ROW_ID}\n"
    binding: dict[str, Any] = {
        "contract": "invented-source-prefill-service-v1",
        "mode": TRAINING_MODE,
        "row_count": 1,
        "record_id_set_sha256": hashlib.sha256(row_ids.encode("utf-8")).hexdigest(),
        "source_prefill": source_prefill,
        "annotation_methodology": methodology,
    }
    if source_prefill == SOURCE_PREFILL_UNAMBIGUOUS:
        binding.update(
            {
                "source_prefill_policy_version": SOURCE_PREFILL_POLICY_VERSION,
                "source_prefill_rows_digest_basis": (
                    "sha256_of_ordered_row_id_and_safe_source_prefill_dto_jsonl"
                ),
                **PREFILL_HASHES,
            }
        )
    return binding


def _definition(
    *,
    prefill: dict[str, Any] | None,
    binding: dict[str, Any] | None = None,
) -> WorkspaceDefinition:
    source = {
        "record_hash": ROW_ID,
        "split": "train",
        "sender": SENDER,
        "sms": SMS,
        "local_model_proposals": [],
    }
    return WorkspaceDefinition(
        mode=TRAINING_MODE,
        rows=(
            WorkbenchSourceRow(
                row_id=ROW_ID,
                position=0,
                sender=SENDER,
                sms=SMS,
                source_json=exact_json_dumps(source),
                split="train",
                queue_tags=("invented-source-prefill",),
                source_prefill=prefill,
            ),
        ),
        binding=_binding() if binding is None else binding,
        metadata={
            "schema_version": 1,
            "fixture": "invented-synthetic-source-prefill-only",
        },
    )


def _temporary_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo = tmp_path / "invented-source-prefill-repo"
    private_root = repo / "PRIVATE_DATA" / "lfm25"
    private_root.mkdir(parents=True)
    private_root.chmod(0o700)
    monkeypatch.setattr(
        "lfm25.annotation_service.require_private_ignore",
        lambda *_args: None,
    )
    return repo


def _db_path(repo: Path) -> Path:
    return repo / "PRIVATE_DATA" / "lfm25" / "source-prefill.sqlite3"


@pytest.mark.parametrize(
    "sms",
    (
        (
            "INR 100 debited from A/c XX0042 at DEMO MART. "
            "The same raw amount INR 100 appears again."
        ),
        (
            "INR 100 debited from A/c XX0042 at DEMO MART. "
            "An equivalent raw amount Rs. 100.00 appears again."
        ),
    ),
)
def test_raw_duplicate_or_equivalent_amounts_suppress_amount_prefill(sms: str) -> None:
    fields = extract_unambiguous_source_fields(sms)

    assert "amount_decimal" not in fields
    assert "amount_span" not in fields


def test_exact_fraction_and_utf8_byte_span_are_preserved() -> None:
    prefix = "कैफे: "
    sms = prefix + "INR 0042.500 debited on A/c XX0042."

    fields = extract_unambiguous_source_fields(sms)

    assert fields["amount_decimal"] == "42.500"
    assert fields["amount_span"] == {
        "text": "INR 0042.500",
        "start": len(prefix.encode("utf-8")),
        "end": len((prefix + "INR 0042.500").encode("utf-8")),
    }
    encoded = sms.encode("utf-8")
    for key in ("amount_span", "account_span"):
        span = fields[key]
        selected = encoded[span["start"] : span["end"]].decode("utf-8")
        assert selected == span["text"]


def test_prefilter_gate_returns_none_and_safe_dto_never_leaks_reason_or_label() -> None:
    rejected_sms = (
        "[SYNTHETIC DEMO ONLY] Your invented verification code is 654321. Do not share it."
    )

    assert (
        source_prefill_for_sms(
            SENDER,
            rejected_sms,
            source_prefill=SOURCE_PREFILL_UNAMBIGUOUS,
        )
        is None
    )
    assert source_prefill_for_sms(SENDER, SMS, source_prefill=SOURCE_PREFILL_OFF) is None

    suggestion = source_prefill_for_sms(
        SENDER,
        SMS,
        source_prefill=SOURCE_PREFILL_UNAMBIGUOUS,
    )
    assert suggestion is not None
    assert set(suggestion).issubset(SAFE_PREFILL_KEYS)
    assert set(suggestion).isdisjoint(FORBIDDEN_PREFILL_KEYS)
    assert suggestion["policy_version"] == SOURCE_PREFILL_POLICY_VERSION


def test_get_draft_complete_and_qc_keep_prefill_separate_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition(prefill=_prefill())

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="invented-prefill-reviewer",
        )

        initial = service.get_row(ROW_ID)
        assert initial["status"] == "pending"
        assert initial["revision"] == 0
        assert initial["annotation"] == empty_annotation()
        assert initial["source_prefill"] == _prefill()
        assert set(initial["source_prefill"]) == SAFE_PREFILL_KEYS
        assert service.history(ROW_ID) == []

        initial["revision"] = 99
        initial["annotation"]["decision"] = "transaction"
        initial["source_prefill"]["amount_decimal"] = "999.00"
        refetched = service.get_row(ROW_ID)
        assert refetched["revision"] == 0
        assert refetched["annotation"]["decision"] is None
        assert refetched["source_prefill"]["amount_decimal"] == "42.50"

        draft_annotation = empty_annotation()
        draft_annotation["notes"] = "Invented draft entered by the reviewer."
        draft = service.save(
            row_id=ROW_ID,
            expected_revision=0,
            annotation=draft_annotation,
            submit=False,
        )["row"]
        assert draft["status"] == "draft"
        assert draft["revision"] == 1
        assert draft["annotation"]["decision"] is None
        assert draft["annotation"]["amount_decimal"] is None
        assert draft["source_prefill"] == _prefill()

        draft_history = service.history(ROW_ID)
        assert [(event["phase"], event["status"]) for event in draft_history] == [
            ("draft", "draft")
        ]
        draft_history[0]["annotation"]["notes"] = "mutated outside the store"
        assert service.history(ROW_ID)[0]["annotation"]["notes"] == (
            "Invented draft entered by the reviewer."
        )

        completed = service.save(
            row_id=ROW_ID,
            expected_revision=1,
            annotation=_annotation(),
            submit=True,
        )["row"]
        assert completed["status"] == "completed"
        assert completed["revision"] == 2
        assert completed["annotation"] == _annotation()
        assert "source_prefill" not in completed
        assert [event["phase"] for event in service.history(ROW_ID)] == [
            "draft",
            "initial",
        ]

        service.start_qc()
        qc_row = service.get_row(ROW_ID)
        assert qc_row["status"] == "completed"
        assert qc_row["revision"] == 2
        assert qc_row["qc_required"] is True
        assert qc_row["qc_status"] == "pending"
        assert qc_row["annotation"] == empty_annotation()
        assert qc_row["history_available"] is False
        assert qc_row["sender"] == SENDER
        assert qc_row["sms"] == SMS
        assert "source_prefill" not in qc_row
        assert service.history(ROW_ID) == []
        assert [event["phase"] for event in store.get_history(ROW_ID)] == [
            "draft",
            "initial",
        ]


@pytest.mark.parametrize("forbidden_key", ("decision", "counterparty_absent"))
def test_service_rejects_forbidden_source_prefill_keys(
    forbidden_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    tampered = _prefill()
    tampered[forbidden_key] = "transaction" if forbidden_key == "decision" else False
    definition = _definition(prefill=tampered)

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        with pytest.raises(
            WorkbenchError,
            match="frozen source-prefill suggestion is invalid",
        ):
            AnnotationService(
                repo_root=repo,
                definition=definition,
                store=store,
                reviewer="invented-prefill-reviewer",
            )


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        (
            _binding(
                source_prefill=SOURCE_PREFILL_OFF,
                methodology=HUMAN_VERIFIED_METHODOLOGY,
            ),
            "source-prefill rows require an assisted workspace binding",
        ),
        (
            _binding(methodology=HUMAN_VERIFIED_METHODOLOGY),
            "source-prefill methodology binding is invalid",
        ),
    ),
)
def test_service_rejects_off_or_wrong_methodology_binding(
    binding: dict[str, Any],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition(prefill=_prefill(), binding=binding)

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        with pytest.raises(WorkbenchError, match=message):
            AnnotationService(
                repo_root=repo,
                definition=definition,
                store=store,
                reviewer="invented-prefill-reviewer",
            )


def test_frozen_workspace_rejects_off_binding_and_prefill_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    assisted = _definition(prefill=_prefill())
    database = _db_path(repo)

    with WorkbenchStore(database, workspace_binding=assisted.binding) as store:
        AnnotationService(
            repo_root=repo,
            definition=assisted,
            store=store,
            reviewer="invented-prefill-reviewer",
        )

    off_binding = _binding(
        source_prefill=SOURCE_PREFILL_OFF,
        methodology=HUMAN_VERIFIED_METHODOLOGY,
    )
    with pytest.raises(
        WorkbenchBindingError,
        match="annotation workspace binding does not match",
    ):
        with WorkbenchStore(database, workspace_binding=off_binding):
            pass

    tampered = _prefill()
    tampered.pop("counterparty_span")
    tampered_definition = _definition(prefill=tampered, binding=assisted.binding)
    with WorkbenchStore(database, workspace_binding=assisted.binding) as store:
        with pytest.raises(
            WorkbenchBindingError,
            match="annotation workspace source binding does not match",
        ):
            AnnotationService(
                repo_root=repo,
                definition=tampered_definition,
                store=store,
                reviewer="invented-prefill-reviewer",
            )


def test_assisted_training_export_records_only_frozen_prefill_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition(prefill=_prefill())
    output = repo / "PRIVATE_DATA" / "lfm25" / "invented-assisted.jsonl"
    report_path = repo / "PRIVATE_DATA" / "lfm25" / "invented-assisted-report.json"

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="invented-prefill-reviewer",
        )
        service.save(
            row_id=ROW_ID,
            expected_revision=0,
            annotation=_annotation(),
            submit=True,
        )
        report = service.export_training(
            output_file=output,
            report_file=report_path,
        )

    exported = exact_json_loads(output.read_text(encoding="utf-8").splitlines()[0])
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report == report
    assert "source_prefill" not in exported

    workbench = exported["human_annotation_workbench"]
    expected = {
        "annotation_methodology": ASSISTED_METHODOLOGY,
        "source_prefill": SOURCE_PREFILL_UNAMBIGUOUS,
        "source_prefill_policy_version": SOURCE_PREFILL_POLICY_VERSION,
        **PREFILL_HASHES,
    }
    for key, value in expected.items():
        assert report[key] == value
        assert workbench[key] == value
    assert "source_prefill_rows_digest_basis" not in report
    assert "source_prefill_rows_digest_basis" not in workbench
    assert set(workbench["annotation"]).isdisjoint({"source_prefill"})
