"""Synthetic HTTP smoke tests for the loopback-only workbench UI."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from pocketfinancer_sms.analyzer import DeterministicSmsAnalyzer
from pocketfinancer_sms.corpus.grouping import build_grouping
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.triage import evaluate_triage
from pocketfinancer_sms.workbench.service import WorkbenchService
from pocketfinancer_sms.workbench.store import WorkbenchStore
from pocketfinancer_sms.workbench.web import WorkbenchWebServer


def _server(tmp_path: Path) -> tuple[WorkbenchWebServer, str]:
    source_id = "src_" + "b" * 32
    body = "INR 15 was debited from account **1234 at SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india"))
    ).analyze(body, operation_id=source_id, is_outgoing=False)
    triage = evaluate_triage(analysis)
    grouping = build_grouping(b"web-synthetic-key" * 2, body, "SYNTH-BANK", "2024-02-01T00:00:00Z")
    record = {
        "contract": "pocketfinancer.corpus-record/1",
        "source_id": source_id,
        "source": {"body": body, "sender": "SYNTH-BANK"},
        "source_metadata": {
            "source_record_id": "synthetic-web",
            "source_row_index": 0,
            "timestamp": "2024-02-01T00:00:00Z",
            "service": "SMS",
            "is_outgoing": False,
        },
        "analysis": analysis.to_dict(),
        "weak_facets": {
            "disposition": triage.disposition.value,
            "selector_action": triage.selector_action.value,
            "operational_class": "posted_candidate",
            "event_state": "posted",
            "financial_family": "card_purchase",
            "payment_rail": "card",
            "confidence": "medium",
            "reason_codes": list(triage.reason_codes),
        },
        "grouping": asdict(grouping),
        "pool": "protected_test",
        "review_state": "unreviewed",
        "provenance": {"corpus_run_id": "synthetic-web-run"},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    store = WorkbenchStore(tmp_path / "private" / "workbench.sqlite3")
    store.import_manifest(manifest, corpus_run_id="synthetic-web-run")
    server = WorkbenchWebServer(
        WorkbenchService(store),
        port=0,
        token="synthetic-token",
        backup_root=tmp_path / "backups",
        export_root=tmp_path / "exports",
    )
    return server, source_id


def _request(
    server: WorkbenchWebServer,
    path: str,
    *,
    authorized: bool = True,
    payload: dict | None = None,
) -> urllib.request.Request:
    headers = {}
    if authorized:
        headers["X-Workbench-Token"] = server.token
        headers["Origin"] = f"http://127.0.0.1:{server.port}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )


def test_synthetic_ui_smoke_is_token_protected_local_and_blind_first(tmp_path: Path) -> None:
    server, source_id = _server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/?token={server.token}"
        ) as response:
            index = response.read().decode("utf-8")
            assert "PocketFinancer SMS Review" in index
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            assert "https://" not in index

        try:
            urllib.request.urlopen(
                _request(server, "/api/progress?reviewer_id=synthetic-reviewer", authorized=False)
            )
            raise AssertionError("unauthorized request unexpectedly succeeded")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        with urllib.request.urlopen(
            _request(
                server,
                f"/api/row?reviewer_id=synthetic-reviewer&source_id={source_id}",
            )
        ) as response:
            row = json.loads(response.read())
            assert row["blind_locked"] is True
            assert "analysis" not in row
            assert row["source"]["body"].startswith("INR")

        with urllib.request.urlopen(
            _request(
                server,
                "/api/draft",
                payload={
                    "source_id": source_id,
                    "reviewer_id": "synthetic-reviewer",
                    "expected_revision": 0,
                    "payload": {"decision": "posted"},
                },
            )
        ) as response:
            saved = json.loads(response.read())
            assert saved["revision"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_non_loopback_bind_is_rejected(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path / "private" / "workbench.sqlite3")
    try:
        WorkbenchWebServer(WorkbenchService(store), host="0.0.0.0", port=0)
        raise AssertionError("non-loopback bind unexpectedly succeeded")
    except ValueError as exc:
        assert "127.0.0.1" in str(exc)
