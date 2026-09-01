"""Token-protected loopback HTTP surface for the local workbench."""

from __future__ import annotations

import json
import mimetypes
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .service import WorkbenchService, WorkbenchValidationError
from .store import WorkbenchConflict


MAX_REQUEST_BYTES = 1_000_000
ASSET_ROOT = Path(__file__).with_name("assets")


class WorkbenchWebServer:
    def __init__(
        self,
        service: WorkbenchService,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        backup_root: Path | None = None,
        export_root: Path | None = None,
        token: str | None = None,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("workbench server may bind only to 127.0.0.1")
        self.service = service
        self.host = host
        self.port = port
        self.backup_root = backup_root
        self.export_root = export_root
        self.token = token or secrets.token_urlsafe(32)
        self._server = ThreadingHTTPServer((host, port), self._handler_type())
        self.port = int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def close(self) -> None:
        self._server.server_close()

    def _handler_type(self):
        service = self.service
        token = self.token
        expected_origin = lambda handler: f"http://{handler.server.server_address[0]}:{handler.server.server_address[1]}"
        backup_root = self.backup_root
        export_root = self.export_root

        class Handler(BaseHTTPRequestHandler):
            server_version = "PocketFinancerWorkbench"
            sys_version = ""

            def log_message(self, _format: str, *_args: object) -> None:
                # Request paths can contain private search terms. Never log them.
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path.startswith("/api/"):
                    if not self._authorized(expected_origin(self)):
                        return
                    self._api_get(parsed.path, parse_qs(parsed.query))
                    return
                query = parse_qs(parsed.query)
                if parsed.path == "/" and query.get("token") != [token]:
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "workbench token required"})
                    return
                asset = "index.html" if parsed.path == "/" else parsed.path.removeprefix("/assets/")
                self._asset(asset)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if not parsed.path.startswith("/api/") or not self._authorized(expected_origin(self)):
                    return
                try:
                    payload = self._read_json()
                    if parsed.path == "/api/draft":
                        result = service.save_draft(**payload)
                    elif parsed.path == "/api/submit":
                        result = service.submit(**payload)
                    elif parsed.path == "/api/adjudicate":
                        result = service.submit(**payload, adjudicated=True)
                    elif parsed.path == "/api/reveal":
                        result = service.reveal(payload["source_id"], payload["reviewer_id"])
                    elif parsed.path == "/api/correction":
                        result = service.correct_weak_facets(**payload)
                    elif parsed.path == "/api/backup" and backup_root is not None:
                        result = service.store.create_backup(backup_root)
                    elif parsed.path == "/api/export" and export_root is not None:
                        result = service.store.export_labels(export_root)
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
                        return
                except WorkbenchConflict as exc:
                    self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                    return
                except (WorkbenchValidationError, KeyError, TypeError, ValueError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except Exception:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "local workbench operation failed"},
                    )
                    return
                self._json(HTTPStatus.OK, result)

            def _api_get(self, path: str, query: dict[str, list[str]]) -> None:
                try:
                    reviewer = _one(query, "reviewer_id", required=True)
                    if path == "/api/progress":
                        result = service.store.progress()
                    elif path == "/api/rows":
                        filters = {
                            name: _one(query, name)
                            for name in (
                                "pool",
                                "operational_class",
                                "event_state",
                                "financial_family",
                                "payment_rail",
                                "sender_template_group",
                                "time_group",
                                "disposition",
                                "selector_action",
                                "review_state",
                            )
                        }
                        result = service.list_rows(
                            reviewer_id=reviewer,
                            filters=filters,
                            search=_one(query, "search"),
                            sort=_one(query, "sort") or "timestamp",
                            descending=(_one(query, "descending") == "true"),
                            limit=int(_one(query, "limit") or 50),
                            offset=int(_one(query, "offset") or 0),
                        )
                    elif path == "/api/row":
                        result = service.view_row(
                            _one(query, "source_id", required=True), reviewer
                        )
                    elif path == "/api/preview":
                        result = service.target_preview(
                            _one(query, "source_id", required=True), reviewer
                        )
                    elif path == "/api/disagreements":
                        result = service.disagreements(
                            _one(query, "source_id", required=True)
                        )
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
                        return
                except (WorkbenchValidationError, KeyError, TypeError, ValueError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except Exception:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "local workbench query failed"},
                    )
                    return
                self._json(HTTPStatus.OK, result)

            def _authorized(self, origin: str) -> bool:
                supplied = self.headers.get("X-Workbench-Token")
                request_origin = self.headers.get("Origin")
                if supplied != token or (request_origin is not None and request_origin != origin):
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "workbench authorization failed"})
                    return False
                return True

            def _read_json(self) -> dict[str, Any]:
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("application/json"):
                    raise ValueError("application/json request required")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("request length is invalid") from exc
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request size is invalid")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("request JSON must be an object")
                return value

            def _asset(self, name: str) -> None:
                if name not in {"index.html", "app.js", "styles.css"}:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
                    return
                path = ASSET_ROOT / name
                try:
                    payload = path.read_bytes()
                except OSError:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "asset unavailable"})
                    return
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"{content_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", _csp())
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(payload)

            def _json(self, status: HTTPStatus, value: Any) -> None:
                payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", _csp())
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(payload)

        return Handler


def _one(query: dict[str, list[str]], name: str, *, required: bool = False) -> str | None:
    values = query.get(name, [])
    if len(values) > 1:
        raise ValueError("query parameter may appear only once")
    value = values[0] if values else None
    if required and not value:
        raise ValueError(f"{name} is required")
    return value


def _csp() -> str:
    return "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
