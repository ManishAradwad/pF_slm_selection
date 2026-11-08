"""Invented supplied-response mechanics only; no dataset/model quality claim."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import localcontext
from pathlib import Path
from threading import Barrier

import pytest

from pocketfinancer_sms import stateful_evaluation
from pocketfinancer_sms.admission import admission_action
from pocketfinancer_sms.currency import MAX_SIGNED_64, parse_money
from pocketfinancer_sms.types import CurrencyProvenance
from pocketfinancer_sms.stateful_evaluation import (
    InventedMessage,
    SuppliedRuntime,
    SyntheticPipelineRun,
)

BODY = "INR 17.25 was debited from account XX7654."


def message(source_id: str | None = "invented-1", *, body: str = BODY, time: int = 1000):
    return InventedMessage(body, "SYNTHETIC", time,
                           "invented-connector" if source_id is not None else None, source_id)


def posted(body: str = BODY) -> str:
    def span(text: str):
        start = body.index(text)
        return {"start_scalar": start, "end_scalar": start + len(text), "text": text}
    return json.dumps({
        "decision": "posted",
        "amount": {"value": "17.25", "currency": "INR", "evidence": span("INR 17.25")},
        "direction": {"value": "debit", "evidence": span("debited")},
        "account": {"reference": "XX7654", "evidence": span("XX7654")},
        "counterparty": None,
    })


def test_shared_admission_vectors():
    path = Path(__file__).parent / "golden/admission-v1.json"
    vectors = json.loads(path.read_text())
    assert vectors["contract"] == "same-source-admission/1"
    for case in vectors["cases"]:
        values = {key: value for key, value in case.items() if key not in {"id", "action"}}
        assert admission_action(**values) == case["action"], case["id"]


def test_retry_requires_existing_unchanged_source():
    with pytest.raises(ValueError, match="retry requires"):
        admission_action(reliable_identity=True, existing_state=None, owner_retry=True)
    with pytest.raises(ValueError, match="retry requires"):
        admission_action(reliable_identity=True, existing_state="settled", same_evidence=False,
                         owner_retry=True)


def test_exclusion_and_replay_never_call_runtime(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    sms = message(body="Your OTP is 6824. Do not share it.")
    receipt = run.admit(sms)
    runtime = SuppliedRuntime([])
    assert receipt.route == "excluded"
    assert run.execute(receipt.operation_id, runtime).action == "reuse"
    assert run.admit(sms).operation_id == receipt.operation_id
    assert runtime.calls == 0
    assert run.mechanics_summary() == {
        "inputs": 2, "operations": 1, "model_calls": 0, "excluded_inputs": 2,
        "reused_inputs": 1, "transactions": 0, "review_operations": 0,
    }
    assert run.db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 0
    run.close()


def test_unknown_account_confirm_next_auto_save_and_replay(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    runtime = SuppliedRuntime([posted(), posted()])
    first = run.admit(message())
    assert run.execute(first.operation_id, runtime).route == "review"
    assert run.admit(message()).action == "reuse"
    assert runtime.calls == 1
    run.confirm_account(first.operation_id, "invented-account")
    # Explicit owner confirmation is not scored as a model success.
    assert run.admit(message()).route == "transactions"
    second = run.admit(message("invented-2", time=2000))
    assert run.execute(second.operation_id, runtime).route == "transactions"
    assert run.execute(second.operation_id, runtime).action == "reuse"
    assert run.admit(message("invented-2", time=2000)).action == "reuse"
    assert runtime.calls == 2
    ledger = run.db.execute("SELECT payload FROM ledger ORDER BY rowid").fetchall()
    assert len(ledger) == 2
    assert all(json.loads(item[0])["semantic_result"]["money"]["minor_units"] == 1725 for item in ledger)
    assert run.mechanics_summary()["model_calls"] == 2
    run.close()


def test_ready_snapshot_survives_resume_and_changed_configuration_rejected(tmp_path):
    path = tmp_path / "run.sqlite"
    run = SyntheticPipelineRun(path)
    first = run.admit(message())
    settings = run.db.execute("SELECT settings FROM operations").fetchone()[0]
    run.close()
    with pytest.raises(ValueError, match="configuration mismatch"):
        SyntheticPipelineRun(path, primary_currency="USD", enabled_profiles=("core-en",), resume=True)
    resumed = SyntheticPipelineRun(path, resume=True)
    replay = resumed.admit(message())
    assert replay.action == "resume" and replay.operation_id == first.operation_id
    assert resumed.db.execute("SELECT settings FROM operations").fetchone()[0] == settings
    runtime = SuppliedRuntime(['{"decision":"none"}'])
    assert resumed.execute(replay.operation_id, runtime).route == "no_transaction"
    assert resumed.admit(message()).action == "reuse"
    assert runtime.calls == 1
    resumed.close()


def test_fresh_refuses_existing_and_resume_refuses_partial(tmp_path):
    path = tmp_path / "run.sqlite"
    run = SyntheticPipelineRun(path)
    run.close()
    with pytest.raises(FileExistsError):
        SyntheticPipelineRun(path)
    path2 = tmp_path / "partial.sqlite"
    path2.touch()
    with pytest.raises(Exception, match="no such table"):
        SyntheticPipelineRun(path2, resume=True)
    with pytest.raises(ValueError, match="existing regular"):
        SyntheticPipelineRun(tmp_path / "missing.sqlite", resume=True)


def test_same_text_without_reliable_identity_remains_two_inputs(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    first = run.admit(message(None))
    second = run.admit(message(None))
    third = run.admit(message("different-id"))
    assert len({first.operation_id, second.operation_id, third.operation_id}) == 3
    run.close()


def test_same_id_changed_evidence_fails_closed_without_overwrite(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    original = run.admit(message())
    with pytest.raises(ValueError, match="immutable evidence"):
        run.admit(message(body=BODY + " Extra content."))
    assert run.admit(message()).operation_id == original.operation_id
    assert run.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 1
    run.close()


def test_concurrent_admission_uses_one_durable_operation(tmp_path):
    path = tmp_path / "run.sqlite"
    SyntheticPipelineRun(path).close()
    def admit_one(_index):
        run = SyntheticPipelineRun(path, resume=True)
        try:
            return run.admit(message()).operation_id
        finally:
            run.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(admit_one, range(4)))
    assert len(set(ids)) == 1


def test_busy_replay_and_recovery_never_start_another_model(tmp_path):
    path = tmp_path / "run.sqlite"
    run = SyntheticPipelineRun(path)
    operation = run.admit(message()).operation_id
    # Invented crash after claim: a second instance may not auto-steal it.
    run.db.execute("UPDATE operations SET state = 'in_flight' WHERE id = ?", (operation,))
    other = SyntheticPipelineRun(path, resume=True)
    runtime = SuppliedRuntime([])
    assert other.admit(message()).action == "in_flight"
    assert other.execute(operation, runtime).action == "in_flight"
    assert runtime.calls == 0
    assert other.recover_interrupted() == 1
    assert other.admit(message()).route == "retryable"
    assert other.execute(operation, runtime).action == "reuse"
    child = other.admit(message(), retry_of=operation)
    assert child.action == "retry" and child.operation_id != operation
    assert other.db.execute("SELECT parent_id FROM operations WHERE id = ?",
                            (child.operation_id,)).fetchone()[0] == operation
    assert other.execute(child.operation_id, SuppliedRuntime(['{"decision":"none"}'])).route == "no_transaction"
    assert other.admit(message()).operation_id == child.operation_id
    with pytest.raises(ValueError, match="latest"):
        other.admit(message(), retry_of=operation)
    run.close()
    other.close()


@pytest.mark.parametrize("response", ['{"decision":"abstain"}', 'not json', RuntimeError("fake")])
def test_failures_are_review_not_false_none(tmp_path, response):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    receipt = run.admit(message())
    runtime = SuppliedRuntime([response])
    assert run.execute(receipt.operation_id, runtime).route == "review"
    assert run.admit(message()).action == "reuse"
    assert runtime.calls == 1
    assert run.db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 0
    run.close()


def test_confirmation_rolls_back_account_when_ledger_conflicts(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    receipt = run.admit(message())
    run.execute(receipt.operation_id, SuppliedRuntime([posted()]))
    # Inject a storage failure, not fake an owner/model success.
    run.db.execute("CREATE TRIGGER fail_save BEFORE INSERT ON ledger BEGIN SELECT RAISE(ABORT, 'invented failure'); END")
    with pytest.raises(Exception, match="invented failure"):
        run.confirm_account(receipt.operation_id, "invented-account")
    assert run.db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0
    assert run.admit(message()).route == "review"
    run.close()


def test_confirmed_fingerprint_flags_same_time_different_alert(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    first = run.admit(message())
    run.execute(first.operation_id, SuppliedRuntime([posted()]))
    run.confirm_account(first.operation_id, "invented-account")
    second = run.admit(message("invented-2"))
    runtime = SuppliedRuntime([posted()])
    assert run.execute(second.operation_id, runtime).route == "review"
    payload = json.loads(run.db.execute(
        "SELECT payload FROM operations WHERE id = ?", (second.operation_id,)
    ).fetchone()[0])
    assert "duplicate_possible" in payload["reason_codes"]
    assert run.mechanics_summary()["transactions"] == 1
    run.close()


def test_confirmation_rechecks_current_resolved_account_duplicates(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    held = run.admit(message())
    run.execute(held.operation_id, SuppliedRuntime([posted()]))
    seed = run.admit(message("seed", time=2000))
    run.execute(seed.operation_id, SuppliedRuntime([posted()]))
    run.confirm_account(seed.operation_id, "invented-account")
    competing = run.admit(message("competing"))
    assert run.execute(competing.operation_id, SuppliedRuntime([posted()])).route == "transactions"
    with pytest.raises(ValueError, match="possible duplicate"):
        run.confirm_account(held.operation_id, "invented-account")
    assert run.admit(message()).route == "review"
    assert run.mechanics_summary()["transactions"] == 2
    run.close()


def test_concurrent_settlement_rechecks_fingerprint_under_write_lock(tmp_path):
    path = tmp_path / "run.sqlite"
    run = SyntheticPipelineRun(path)
    seed = run.admit(message())
    run.execute(seed.operation_id, SuppliedRuntime([posted()]))
    run.confirm_account(seed.operation_id, "invented-account")
    run.close()
    barrier = Barrier(2)

    class SynchronizedRuntime(SuppliedRuntime):
        def __call__(self, request, cancellation):
            barrier.wait(timeout=10)
            return super().__call__(request, cancellation)

    def execute_one(source_id):
        lane = SyntheticPipelineRun(path, resume=True)
        try:
            receipt = lane.admit(message(source_id, time=2000))
            runtime = SynchronizedRuntime([posted()])
            result = lane.execute(receipt.operation_id, runtime)
            return result.route, runtime.calls
        finally:
            lane.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute_one, ["race-a", "race-b"]))
    assert sorted(route for route, _calls in results) == ["review", "transactions"]
    assert [calls for _route, calls in results] == [1, 1]
    resumed = SyntheticPipelineRun(path, resume=True)
    assert resumed.mechanics_summary()["transactions"] == 2
    assert resumed.mechanics_summary()["model_calls"] == 3
    resumed.close()


def test_retry_fences_stale_ready_parent_without_second_runtime_call(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    parent = run.admit(message())
    child = run.admit(message(), retry_of=parent.operation_id)
    runtime = SuppliedRuntime(['{"decision":"none"}'])
    stale = run.execute(parent.operation_id, runtime)
    assert stale.route == "superseded" and runtime.calls == 0
    assert run.execute(child.operation_id, runtime).route == "no_transaction"
    assert runtime.calls == 1
    run.close()


@pytest.mark.parametrize("dependency", ["structural_text.py", "profiles.py"])
def test_resume_rejects_changed_reachable_dependency(tmp_path, monkeypatch, dependency):
    path = tmp_path / "run.sqlite"
    SyntheticPipelineRun(path).close()
    hashes = stateful_evaluation._host_source_hashes()
    hashes[dependency] = "0" * 64
    monkeypatch.setattr(stateful_evaluation, "_host_source_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="configuration mismatch"):
        SyntheticPipelineRun(path, resume=True)


def test_oversized_amount_is_validation_review_not_interrupted(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    huge = "9" * 40
    body = f"INR {huge} was debited from account XX7654."
    response = json.loads(posted())
    start = body.index("INR")
    response["amount"] = {
        "value": huge, "currency": "INR",
        "evidence": {"start_scalar": start, "end_scalar": start + 4 + len(huge),
                     "text": "INR " + huge},
    }
    for field, text in [("direction", "debited"), ("account", "XX7654")]:
        offset = body.index(text)
        response[field]["evidence"] = {
            "start_scalar": offset, "end_scalar": offset + len(text), "text": text,
        }
    receipt = run.admit(message(body=body))
    assert run.execute(receipt.operation_id, SuppliedRuntime([json.dumps(response)])).route == "review"
    assert run.db.execute("SELECT state FROM operations").fetchone()[0] == "settled"
    assert run.mechanics_summary()["transactions"] == 0
    run.close()


def test_exact_signed_64_money_boundary_survives_decimal_context():
    provenance = CurrencyProvenance.EXPLICIT_CODE
    with localcontext() as context:
        context.prec = 2
        assert parse_money("92233720368547758.07", currency="USD", provenance=provenance).minor_units == MAX_SIGNED_64
        with pytest.raises(ValueError, match="signed 64-bit"):
            parse_money("92233720368547758.08", currency="USD", provenance=provenance)


def test_malformed_unicode_response_is_captured_and_strictly_validated(tmp_path):
    run = SyntheticPipelineRun(tmp_path / "run.sqlite")
    receipt = run.admit(message())
    raw = "\ud800"
    assert run.execute(receipt.operation_id, SuppliedRuntime([raw])).route == "review"
    captured = run.db.execute("SELECT raw_output FROM attempts").fetchone()[0]
    assert json.loads(captured) == raw
    payload = json.loads(run.db.execute("SELECT payload FROM operations").fetchone()[0])
    assert payload["reason_codes"] == ["extractor_malformed_json"]
    run.close()
