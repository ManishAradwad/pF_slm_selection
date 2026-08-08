from __future__ import annotations

import hashlib
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any

import pytest

from lfm25.annotation_service import AnnotationService
from lfm25.annotation_store import WorkbenchStore
from lfm25.annotation_web import (
    COOKIE_NAME,
    LOOPBACK_HOST,
    WorkbenchHTTPServer,
    _available_filters,
)
from lfm25.annotation_workbench import (
    ACTIVE_LEARNING_QUEUE_POLICY_VERSION,
    ACTIVE_LEARNING_QUEUE_TAGS,
    ASSISTED_METHODOLOGY,
    BLINDED_MODE,
    FILTER_NAMES,
    SOURCE_PREFILL_UNAMBIGUOUS,
    TRAINING_MODE,
    WorkspaceDefinition,
    WorkbenchSourceRow,
    empty_annotation,
    exact_json_dumps,
    exact_json_loads,
)


WEB_SMS = "Synthetic debit of INR 12.50 from acct X."


def _definition() -> WorkspaceDefinition:
    row_id = "synthetic-web-1"
    amount_text = "12.50"
    amount_start = len(WEB_SMS[: WEB_SMS.index(amount_text)].encode("utf-8"))
    account_text = "acct X"
    account_start = len(WEB_SMS[: WEB_SMS.index(account_text)].encode("utf-8"))
    source_prefill = {
        "policy_version": 1,
        "amount_decimal": amount_text,
        "amount_span": {
            "text": amount_text,
            "start": amount_start,
            "end": amount_start + len(amount_text.encode("utf-8")),
        },
        "type": "debit",
        "account_span": {
            "text": account_text,
            "start": account_start,
            "end": account_start + len(account_text.encode("utf-8")),
        },
    }
    return WorkspaceDefinition(
        mode=TRAINING_MODE,
        rows=(
            WorkbenchSourceRow(
                row_id=row_id,
                position=0,
                sender="SYNTH-WEB",
                sms=WEB_SMS,
                source_json=exact_json_dumps(
                    {
                        "record_hash": row_id,
                        "split": "train",
                        "sender": "SYNTH-WEB",
                        "sms": WEB_SMS,
                        "local_model_proposals": [],
                    }
                ),
                split="train",
                queue_tags=("synthetic-web",),
                source_prefill=source_prefill,
            ),
        ),
        binding={
            "contract": "synthetic-annotation-web-v1",
            "mode": TRAINING_MODE,
            "row_count": 1,
            "source_prefill": SOURCE_PREFILL_UNAMBIGUOUS,
            "annotation_methodology": ASSISTED_METHODOLOGY,
            "record_id_set_sha256": hashlib.sha256(f"{row_id}\n".encode()).hexdigest(),
        },
        metadata={"schema_version": 1, "fixture": "invented-web-only"},
    )


def _temporary_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo = tmp_path / "synthetic-web-repo"
    private_root = repo / "PRIVATE_DATA" / "lfm25"
    private_root.mkdir(parents=True)
    private_root.chmod(0o700)
    monkeypatch.setattr(
        "lfm25.annotation_service.require_private_ignore", lambda *_args: None
    )
    monkeypatch.setattr(
        "lfm25.annotation_sources.require_private_ignore", lambda *_args: None
    )
    return repo


def _request(
    server: WorkbenchHTTPServer,
    method: str,
    path: str,
    *,
    body: str | None = None,
    headers: dict[str, str] | None = None,
    host: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection(
        LOOPBACK_HOST,
        int(server.server_address[1]),
        timeout=3,
    )
    connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    connection.putheader(
        "Host",
        host or f"{LOOPBACK_HOST}:{server.server_address[1]}",
    )
    for name, value in (headers or {}).items():
        connection.putheader(name, value)
    payload = None if body is None else body.encode("utf-8")
    if payload is not None:
        connection.putheader("Content-Length", str(len(payload)))
    connection.endheaders(payload)
    response = connection.getresponse()
    response_body = response.read()
    response_headers = dict(response.getheaders())
    status = response.status
    connection.close()
    return status, response_headers, response_body


def _decoded(body: bytes) -> dict[str, Any]:
    value = exact_json_loads(body.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def test_loopback_http_security_valid_save_and_session_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition()
    db_path = repo / "PRIVATE_DATA" / "lfm25" / "annotation_workbench" / "web.sqlite3"
    store = WorkbenchStore(db_path, workspace_binding=definition.binding)
    service = AnnotationService(
        repo_root=repo,
        definition=definition,
        store=store,
        reviewer="synthetic-web-reviewer",
    )
    server = WorkbenchHTTPServer(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == LOOPBACK_HOST == "127.0.0.1"

        bad_host_status, _, bad_host_body = _request(
            server,
            "GET",
            "/",
            host="attacker.invalid",
        )
        assert bad_host_status == HTTPStatus.MISDIRECTED_REQUEST
        assert "Host" in _decoded(bad_host_body)["error"]

        root_status, root_headers, root_body = _request(server, "GET", "/")
        assert root_status == HTTPStatus.OK
        assert root_headers["Content-Type"].startswith("text/html")
        assert "default-src 'none'" in root_headers["Content-Security-Policy"]
        root_markup = root_body.decode("utf-8")
        assert "No SMS loaded" in root_markup
        assert '<div id="app-shell" hidden inert>' in root_markup
        assert WEB_SMS not in root_markup
        cookie = root_headers["Set-Cookie"].split(";", 1)[0]
        assert cookie.startswith(f"{COOKIE_NAME}=")

        assets = (
            ("/assets/app.js", "text/javascript", "trustedLocalRuntime"),
            ("/assets/styles.css", "text/css", ".source-pane"),
        )
        for path, content_type, marker in assets:
            asset_status, asset_headers, asset_body = _request(server, "GET", path)
            assert asset_status == HTTPStatus.OK
            assert asset_headers["Content-Type"].startswith(content_type)
            assert asset_headers["Cache-Control"] == "no-store, max-age=0"
            assert marker in asset_body.decode("utf-8")

        bootstrap_status, _, bootstrap_body = _request(
            server,
            "GET",
            "/api/bootstrap",
            headers={"Cookie": cookie},
        )
        assert bootstrap_status == HTTPStatus.OK
        bootstrap = _decoded(bootstrap_body)
        csrf_token = bootstrap["csrf_token"]
        assert isinstance(csrf_token, str) and csrf_token
        assert "active_learning" in bootstrap["filters"]
        assert bootstrap["mode"] == TRAINING_MODE

        row_status, _, row_body = _request(
            server,
            "GET",
            "/api/row?position=0&direction=first&filter=pending",
            headers={"Cookie": cookie},
        )
        assert row_status == HTTPStatus.OK
        navigated = _decoded(row_body)
        row = navigated["row"]
        assert row["review_id"] == "synthetic-web-1"
        assert row["sender"] == "SYNTH-WEB"
        assert row["sms"] == WEB_SMS
        source_prefill = row["source_prefill"]
        assert set(source_prefill) == {
            "policy_version",
            "amount_decimal",
            "amount_span",
            "type",
            "account_span",
        }
        assert source_prefill["amount_decimal"] == "12.50"
        assert navigated["progress"]["total_rows"] == 1
        assert navigated["progress"]["ready_for_qc"] is False

        draft = empty_annotation()
        draft["notes"] = "Synthetic HTTP draft."
        save_value = {
            "review_id": "synthetic-web-1",
            "expected_revision": 0,
            "annotation": draft,
            "submit": False,
        }
        mutation_headers = {
            "Content-Type": "application/json",
            "Cookie": cookie,
            "X-Workbench-Token": csrf_token,
            "Origin": f"http://{LOOPBACK_HOST}:{server.server_address[1]}",
        }

        bad_csrf_status, _, _ = _request(
            server,
            "POST",
            "/api/save",
            body=exact_json_dumps(save_value),
            headers={**mutation_headers, "X-Workbench-Token": "invalid-token"},
        )
        assert bad_csrf_status == HTTPStatus.FORBIDDEN

        duplicate_json = (
            '{"review_id":"synthetic-web-1",'
            '"review_id":"synthetic-duplicate",'
            '"expected_revision":0,'
            f'"annotation":{exact_json_dumps(draft)},'
            '"submit":false}'
        )
        duplicate_status, _, duplicate_body = _request(
            server,
            "POST",
            "/api/save",
            body=duplicate_json,
            headers=mutation_headers,
        )
        assert duplicate_status == HTTPStatus.BAD_REQUEST
        assert "JSON" in _decoded(duplicate_body)["error"]

        save_status, _, save_body = _request(
            server,
            "POST",
            "/api/save",
            body=exact_json_dumps(save_value),
            headers=mutation_headers,
        )
        assert save_status == HTTPStatus.OK
        saved = _decoded(save_body)
        assert saved["row"]["status"] == "draft"
        assert saved["row"]["revision"] == 1
        stored = store.get_row("synthetic-web-1")
        assert stored["status"] == "draft"
        assert stored["annotation"]["notes"] == "Synthetic HTTP draft."

        close_status, _, close_body = _request(
            server,
            "POST",
            "/api/session/close",
            body="{}",
            headers=mutation_headers,
        )
        assert close_status == HTTPStatus.OK
        assert _decoded(close_body) == {"closed": True}
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=3)
        server.server_close()
        store.close()


def test_ui_serializes_saves_and_flushes_before_qc() -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "lfm25/annotation_assets/app.js"
    )
    script = script_path.read_text(encoding="utf-8")

    assert "saveInFlight: false" in script
    assert "form.inert = busy" in script

    save_section = script.split("async function save", 1)[1].split("function scheduleAutosave", 1)[0]
    assert "state.saveInFlight = true" in save_section
    assert "state.saveInFlight = false" in save_section

    qc_section = script.split("async function startQc", 1)[1].split("async function reveal", 1)[0]
    dirty_flush = qc_section.index("await save(false)")
    qc_request = qc_section.index('api("/api/qc/start"')
    assert dirty_flush < qc_request
    assert qc_section.index("state.loading = true") < qc_request
    assert qc_section.index("setFormBusy(true)") < qc_request

def test_active_learning_filter_contract_is_training_initial_only() -> None:
    expected_tags = (
        "model_disagreement",
        "low_confidence_output",
        "candidate_coverage_miss",
        "hard_negative_with_amount",
        "otp_or_security",
        "pending_failed_declined_or_hold",
        "payment_request_or_reminder",
        "refund_or_reversal",
        "multiple_entities",
        "rare_sender_or_template",
    )
    assert ACTIVE_LEARNING_QUEUE_POLICY_VERSION == 1
    assert ACTIVE_LEARNING_QUEUE_TAGS == expected_tags
    assert FILTER_NAMES.count("active_learning") == 1

    training_filters = _available_filters(mode=TRAINING_MODE, session_phase="initial")
    blinded_filters = _available_filters(mode=BLINDED_MODE, session_phase="initial")
    qc_filters = _available_filters(mode=TRAINING_MODE, session_phase="qc")

    assert "active_learning" in training_filters
    assert "active_learning" not in blinded_filters
    assert "active_learning" not in qc_filters
    assert {"noted", "transaction", "null"}.isdisjoint(qc_filters)


def test_ui_guards_post_label_details_and_async_row_responses() -> None:
    asset_root = Path(__file__).resolve().parents[1] / "lfm25/annotation_assets"
    markup = (asset_root / "index.html").read_text(encoding="utf-8")
    script = (asset_root / "app.js").read_text(encoding="utf-8")

    assert 'value="active_learning" hidden disabled' in markup
    assert 'id="queue-tags-panel"' in markup
    assert 'id="queue-tags"' in markup
    assert 'id="history-panel" class="hidden"' in markup

    queue_section = script.split("function renderQueueTags", 1)[1].split(
        "function spanSummary",
        1,
    )[0]
    assert 'row?.status === "completed"' in queue_section
    assert "Array.isArray(row.queue_tags)" in queue_section
    assert "item.textContent" in queue_section

    history_renderer = script.split("function renderHistoryEvent", 1)[1].split(
        "function renderRow",
        1,
    )[0]
    for component in (
        "event.annotation",
        "annotation.amount_span",
        "annotation.account_span",
        "annotation.counterparty_span",
        "annotation.notes",
    ):
        assert component in history_renderer

    history_loader = script.split("async function loadHistory", 1)[1].split(
        "async function navigate",
        1,
    )[0]
    assert history_loader.count("state.row.review_id !== reviewId") >= 2
    assert history_loader.count("state.row.history_available !== true") >= 2
    assert "renderHistoryEvent(event)" in history_loader

    proposal = script.split("async function revealProposals", 1)[1].split(
        "function editableTarget",
        1,
    )[0]
    for guard in (
        "state.dirty",
        "state.saveInFlight",
        "state.proposalInFlight",
        "state.row.review_id !== reviewId",
    ):
        assert guard in proposal
    assert proposal.count("state.loading") >= 2
    assert "const reviewId = state.row.review_id" in proposal
    response_guard = proposal.index("state.row.review_id !== reviewId")
    panel_reveal = proposal.index('byId("proposal-panel").classList.remove("hidden")')
    assert response_guard < panel_reveal

    autosave = script.split("function scheduleAutosave", 1)[1].split(
        "function configurePhase",
        1,
    )[0]
    assert 'byId("proposal-panel").classList.add("hidden")' in autosave


def test_ui_launch_gate_navigation_qc_and_prefill_contract() -> None:
    asset_root = Path(__file__).resolve().parents[1] / "lfm25/annotation_assets"
    markup = (asset_root / "index.html").read_text(encoding="utf-8")
    styles = (asset_root / "styles.css").read_text(encoding="utf-8")
    script = (asset_root / "app.js").read_text(encoding="utf-8")

    assert "No SMS loaded" in markup
    assert "run_lfm25_annotation_workbench.py blinded" in markup
    assert '<div id="app-shell" hidden inert>' in markup
    assert 'id="sms" class="sms"' in markup
    assert 'id="start-qc" disabled aria-describedby="qc-availability"' in markup
    assert "Suggested locally from deterministic extraction" in markup

    runtime = script.split("function trustedLocalRuntime", 1)[1].split(
        "async function api",
        1,
    )[0]
    assert 'window.location.protocol === "http:"' in runtime
    assert 'window.location.hostname === "127.0.0.1"' in runtime
    assert 'shell.removeAttribute("inert")' in runtime

    navigation = script.split("async function navigate", 1)[1].split(
        "async function save",
        1,
    )[0]
    for behavior in (
        "let loaded = false",
        'byId("filter").value = filter',
        "loaded = true",
        'byId("filter").value = state.filter',
        "return loaded",
    ):
        assert behavior in navigation

    bootstrap = script.split("async function bootstrap", 1)[1].split(
        "bootstrap();",
        1,
    )[0]
    navigate_index = bootstrap.index('await navigate("first", initialFilter)')
    reveal_index = bootstrap.index("revealAppShell()")
    assert navigate_index < reveal_index
    assert "if (!loaded)" in bootstrap
    assert "setLaunchFailure(" in bootstrap

    progress = script.split("function renderProgress", 1)[1].split(
        "function renderQueueTags",
        1,
    )[0]
    assert "progress.ready_for_qc === true" in progress
    assert "button.disabled = !ready || state.qcSession" in progress
    assert 'byId("qc-availability")' in progress

    prefill = script.split("function applySourcePrefill", 1)[1].split(
        "function updateDecisionVisibility",
        1,
    )[0]
    for guard in (
        '["pending", "draft"].includes(row.status)',
        "state.qcSession",
        'selectedRadio("decision") !== "transaction"',
        '!byId("amount-decimal").value.trim()',
        "state.spans.amount === null",
        "state.spans.account === null",
        '!byId("counterparty-absent").checked',
    ):
        assert guard in prefill
    assert "scheduleAutosave" not in prefill

    renderer = script.split("function renderRow", 1)[1].split(
        "async function loadHistory",
        1,
    )[0]
    assert 'selectedRadio("decision") === "transaction"' in renderer
    assert "applySourcePrefill()" in renderer

    manual_amount = script.split("function canonicalAmountFromSelection", 1)[1].split(
        "function renderProgress",
        1,
    )[0]
    assert "matches.length !== 1" in manual_amount
    assert 'byId("amount-decimal").value = canonical' in manual_amount

    wiring = script.split("function wireUi", 1)[1].split(
        "async function bootstrap",
        1,
    )[0]
    assert "recordHumanEdit(input)" in wiring
    assert 'window.addEventListener("beforeunload"' in wiring
    assert "if (!state.dirty) return" in wiring

    assert "grid-template-columns: minmax(0, 1.08fr)" in styles
    assert ".source-pane {" in styles and "position: sticky" in styles
    assert "@media (max-width: 980px)" in styles


def test_amount_and_prefill_helpers_execute_with_invented_sms() -> None:
    node = shutil.which("node") or shutil.which("node.exe")
    if node is None:
        pytest.skip("Node.js is unavailable for the frontend helper contract test.")

    script_path = (
        Path(__file__).resolve().parents[1] / "lfm25/annotation_assets/app.js"
    )
    script = script_path.read_text(encoding="utf-8")
    canonical = "function canonicalAmountFromSelection" + script.split(
        "function canonicalAmountFromSelection",
        1,
    )[1].split("\nfunction useSelection", 1)[0]
    prefill_helpers = "function validPrefillSpan" + script.split(
        "function validPrefillSpan",
        1,
    )[1].split("\nfunction applySourcePrefill", 1)[0]
    program = canonical + "\n" + prefill_helpers + "\n" + r"""
const sms = "Synthetic debit of INR 12.50 from acct X.";
const amountText = "12.50";
const charStart = sms.indexOf(amountText);
const encoder = new TextEncoder();
const start = encoder.encode(sms.slice(0, charStart)).length;
const span = {
  text: amountText,
  start,
  end: start + encoder.encode(amountText).length,
};
const source_prefill = {
  policy_version: 1,
  amount_decimal: amountText,
  amount_span: span,
  type: "debit",
};
const good = sanitizedSourcePrefill({ sms, source_prefill });
const extra = sanitizedSourcePrefill({
  sms,
  source_prefill: { ...source_prefill, decision: "transaction" },
});
const badSpan = sanitizedSourcePrefill({
  sms,
  source_prefill: {
    ...source_prefill,
    amount_span: { ...span, end: span.end + 1 },
  },
});
const policyOnly = sanitizedSourcePrefill({
  sms,
  source_prefill: { policy_version: 1 },
});
const incompleteAmount = sanitizedSourcePrefill({
  sms,
  source_prefill: { policy_version: 1, amount_decimal: amountText },
});
const amounts = [
  canonicalAmountFromSelection("INR 1,23,456.00"),
  canonicalAmountFromSelection("USD 0012.50"),
  canonicalAmountFromSelection("12 and account 34"),
  canonicalAmountFromSelection("-12.00"),
  canonicalAmountFromSelection("- 12.00"),
  canonicalAmountFromSelection("0.00"),
  canonicalAmountFromSelection("12,34"),
];
process.stdout.write(JSON.stringify({
  good, extra, badSpan, policyOnly, incompleteAmount, amounts,
}));
"""
    completed = subprocess.run(
        [node, "-e", program],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    result = exact_json_loads(completed.stdout)
    assert result["good"]["policy_version"] == 1
    assert result["good"]["amount_decimal"] == "12.50"
    assert result["good"]["amount_span"]["text"] == "12.50"
    assert result["good"]["type"] == "debit"
    assert result["extra"] is None
    assert result["badSpan"] is None
    assert result["policyOnly"] is None
    assert result["incompleteAmount"] is None
    assert result["amounts"] == ["123456.00", "12.50", None, None, None, None, None]
