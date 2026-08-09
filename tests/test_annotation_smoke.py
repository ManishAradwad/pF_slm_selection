from __future__ import annotations

from http import HTTPStatus
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_lfm25_annotation_workbench_smoke.py"


def _startup_line(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=10):
            raise AssertionError("synthetic workbench did not report its loopback URL")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise AssertionError(f"synthetic workbench exited before startup: {stderr}")
    return line


def _request(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    connection.putheader("Host", f"127.0.0.1:{port}")
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


def _json(body: bytes) -> dict[str, Any]:
    value = json.loads(body.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def test_synthetic_launcher_serves_real_ui_and_removes_state(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--port",
            "0",
            "--source-prefill",
            "unambiguous",
        ],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        startup_line = _startup_line(process)
        startup = json.loads(startup_line)
        assert startup["synthetic_only"] is True
        assert startup["persistent_state"] is False
        assert startup["raw_values_emitted"] is False
        assert startup["source_prefill"] == "unambiguous"
        assert startup["annotation_methodology"] == (
            "human_verified_candidate_assisted"
        )
        assert "Acme Demo Bank" not in startup_line

        parsed_url = urlsplit(startup["url"])
        assert parsed_url.hostname == "127.0.0.1"
        assert isinstance(parsed_url.port, int)
        port = parsed_url.port

        root_status, root_headers, root_body = _request(port, "GET", "/")
        assert root_status == HTTPStatus.OK
        assert b"Annotation Workbench" in root_body
        cookie = root_headers["Set-Cookie"].split(";", 1)[0]

        for asset_path in ("/assets/styles.css", "/assets/app.js"):
            asset_status, _, asset_body = _request(port, "GET", asset_path)
            assert asset_status == HTTPStatus.OK
            assert asset_body

        bootstrap_status, _, bootstrap_body = _request(
            port,
            "GET",
            "/api/bootstrap",
            headers={"Cookie": cookie},
        )
        assert bootstrap_status == HTTPStatus.OK
        bootstrap = _json(bootstrap_body)
        assert bootstrap["mode"] == "training_curation"
        assert bootstrap["progress"]["total_rows"] == 3

        row_status, _, row_body = _request(
            port,
            "GET",
            "/api/row?position=-1&direction=first&filter=pending",
            headers={"Cookie": cookie},
        )
        assert row_status == HTTPStatus.OK
        row = _json(row_body)["row"]
        assert row["sender"] == "SYNTH-DEMO-BANK"
        assert "SYNTHETIC DEMO ONLY - NO PRIVATE DATA" in row["sms"]
        assert "INR 42.50" in row["sms"]
        assert row["status"] == "pending"
        assert row["revision"] == 0
        assert row["annotation"]["decision"] is None
        assert row["annotation"]["amount_decimal"] is None
        assert row["source_prefill"]["amount_decimal"] == "42.50"
        assert row["source_prefill"]["type"] == "debit"
        assert row["source_prefill"]["amount_span"]["text"] == "INR 42.50"

        history_status, _, history_body = _request(
            port,
            "GET",
            "/api/history?review_id=synthetic-smoke-transaction-debit",
            headers={"Cookie": cookie},
        )
        assert history_status == HTTPStatus.OK
        assert _json(history_body) == {"events": []}

        token = bootstrap["csrf_token"]
        close_status, _, close_body = _request(
            port,
            "POST",
            "/api/session/close",
            headers={
                "Content-Type": "application/json",
                "Cookie": cookie,
                "X-Workbench-Token": token,
                "Origin": f"http://127.0.0.1:{port}",
            },
            body="{}",
        )
        assert close_status == HTTPStatus.OK
        assert _json(close_body) == {"closed": True}
        assert process.wait(timeout=10) == 0
        assert process.stdout is not None
        assert process.stderr is not None
        assert process.stdout.read() == ""
        assert process.stderr.read() == ""
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

    assert list(tmp_path.iterdir()) == []


def test_smoke_launcher_has_no_private_loader_or_export_path() -> None:
    source = SMOKE_SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "load_blinded_workspace",
        "load_training_workspace",
        "private_path",
        "export_training",
    ):
        assert forbidden not in source
