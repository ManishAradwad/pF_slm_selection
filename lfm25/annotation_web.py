"""Localhost-only HTTP surface for the PocketFinancer annotation workbench."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import threading
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from .annotation_service import AnnotationService, safe_service_error
from .annotation_store import StaleRevisionError, WorkbenchStoreError
from .annotation_workbench import (
    BLINDED_MODE,
    FILTER_NAMES,
    TRAINING_MODE,
    AnnotationValidationError,
    WorkbenchError,
    exact_json_dumps,
    exact_json_loads,
)


LOOPBACK_HOST = "127.0.0.1"
MAX_REQUEST_BYTES = 256 * 1024
COOKIE_NAME = "pf_local_workbench"
ASSET_ROOT = Path(__file__).with_name("annotation_assets")
ASSETS = {
    "/": (ASSET_ROOT / "index.html", "text/html; charset=utf-8"),
    "/assets/app.js": (ASSET_ROOT / "app.js", "text/javascript; charset=utf-8"),
    "/assets/styles.css": (ASSET_ROOT / "styles.css", "text/css; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; font-src 'none'; object-src 'none'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
}


def _available_filters(*, mode: str, session_phase: str) -> list[str]:
    """Return only filters that cannot disclose forbidden workflow information."""

    hidden: set[str] = set()
    if mode != TRAINING_MODE or session_phase == "qc":
        hidden.add("active_learning")
    if session_phase == "qc":
        hidden.update({"noted", "transaction", "null"})
    return [name for name in FILTER_NAMES if name not in hidden]


class WorkbenchHTTPServer(HTTPServer):
    """A fixed-loopback server carrying one locked workbench service."""

    allow_reuse_address = False

    def __init__(self, service: AnnotationService, *, port: int = 0) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise WorkbenchError("the local workbench port is invalid")
        self.service = service
        self.capability_token = secrets.token_urlsafe(32)
        super().__init__((LOOPBACK_HOST, port), WorkbenchRequestHandler)

    @property
    def local_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.server_address[1]}/"


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    """Exact route allowlist with content-free access/error handling."""

    server: WorkbenchHTTPServer
    server_version = "PocketFinancerLocal"
    sys_version = ""
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10.0)


    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _host_valid(self) -> bool:
        host = self.headers.get("Host", "")
        return host == f"{LOOPBACK_HOST}:{self.server.server_address[1]}"

    def _cookie_valid(self) -> bool:
        cookie = self.headers.get("Cookie", "")
        values = {}
        for item in cookie.split(";"):
            key, separator, value = item.strip().partition("=")
            if separator:
                values[key] = value
        return secrets.compare_digest(
            values.get(COOKIE_NAME, ""), self.server.capability_token
        )

    def _mutation_valid(self) -> bool:
        supplied = self.headers.get("X-Workbench-Token", "")
        expected_origin = f"http://{LOOPBACK_HOST}:{self.server.server_address[1]}"
        origin = self.headers.get("Origin")
        return (
            self._cookie_valid()
            and secrets.compare_digest(supplied, self.server.capability_token)
            and (origin is None or origin == expected_origin)
        )

    def _start_response(
        self,
        status: HTTPStatus | int,
        *,
        content_type: str,
        length: int,
        set_cookie: bool = False,
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"{COOKIE_NAME}={self.server.capability_token}; HttpOnly; SameSite=Strict; Path=/",
            )
        self.end_headers()

    def _json(
        self,
        status: HTTPStatus | int,
        value: Mapping[str, Any],
        *,
        set_cookie: bool = False,
    ) -> None:
        payload = exact_json_dumps(value).encode("utf-8")
        self._start_response(
            status,
            content_type="application/json; charset=utf-8",
            length=len(payload),
            set_cookie=set_cookie,
        )
        self.wfile.write(payload)

    def _error(
        self,
        status: HTTPStatus | int,
        message: str,
        *,
        problems: list[dict[str, str]] | None = None,
    ) -> None:
        value: dict[str, Any] = {"error": message}
        if problems is not None:
            value["problems"] = problems
        self._json(status, value)

    def _require_host(self) -> bool:
        if self._host_valid():
            return True
        self._error(HTTPStatus.MISDIRECTED_REQUEST, "The local Host header is invalid.")
        return False

    def _require_cookie(self) -> bool:
        if self._cookie_valid():
            return True
        self._error(HTTPStatus.FORBIDDEN, "The local workbench session is unavailable.")
        return False

    def _serve_asset(self, path: str) -> None:
        asset = ASSETS.get(path)
        if asset is None:
            self._error(HTTPStatus.NOT_FOUND, "The local route was not found.")
            return
        file_path, content_type = asset
        try:
            payload = file_path.read_bytes()
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "A local asset is unavailable.")
            return
        self._start_response(
            HTTPStatus.OK,
            content_type=content_type,
            length=len(payload),
            set_cookie=path == "/",
        )
        self.wfile.write(payload)

    @staticmethod
    def _one(query: Mapping[str, list[str]], name: str) -> str:
        values = query.get(name)
        if values is None or len(values) != 1:
            raise WorkbenchError("the local request parameters are invalid")
        return values[0]

    def _bootstrap(self) -> None:
        progress = self.server.service.progress()
        self._json(
            HTTPStatus.OK,
            {
                "contract": "pocketfinancer-local-annotation-workbench-v1",
                "mode": self.server.service.mode,
                "mode_label": (
                    "Blinded Test Adjudication"
                    if self.server.service.mode == BLINDED_MODE
                    else "Training Data Curation"
                ),
                "reviewer": self.server.service.reviewer,
                "session_phase": self.server.service.session_phase,
                "filters": _available_filters(
                    mode=self.server.service.mode,
                    session_phase=self.server.service.session_phase,
                ),
                "batch_size": self.server.service.batch_size,
                "progress": progress,
                "csrf_token": self.server.capability_token,
            },
        )

    def _get_row(self, query: Mapping[str, list[str]]) -> None:
        allowed = {"position", "direction", "filter"}
        if set(query) != allowed:
            raise WorkbenchError("the local request parameters are invalid")
        try:
            position = int(self._one(query, "position"))
        except ValueError as exc:
            raise WorkbenchError("the local request parameters are invalid") from exc
        result = self.server.service.navigate(
            position=position,
            direction=self._one(query, "direction"),
            filter_name=self._one(query, "filter"),
        )
        self._json(HTTPStatus.OK, result)

    def _get_history(self, query: Mapping[str, list[str]]) -> None:
        if set(query) != {"review_id"}:
            raise WorkbenchError("the local request parameters are invalid")
        self._json(
            HTTPStatus.OK,
            {"events": self.server.service.history(self._one(query, "review_id"))},
        )

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_host():
            return
        target = urlsplit(self.path)
        if target.path in ASSETS:
            self._serve_asset(target.path)
            return
        if target.path not in {"/api/bootstrap", "/api/row", "/api/history"}:
            self._error(HTTPStatus.NOT_FOUND, "The local route was not found.")
            return
        if not self._require_cookie():
            return
        try:
            query = parse_qs(target.query, keep_blank_values=True, strict_parsing=True)
            if target.path == "/api/bootstrap":
                if query:
                    raise WorkbenchError("the local request parameters are invalid")
                self._bootstrap()
            elif target.path == "/api/row":
                self._get_row(query)
            else:
                self._get_history(query)
        except AnnotationValidationError as exc:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                str(exc),
                problems=[item.as_dict() for item in exc.problems],
            )
        except (WorkbenchError, WorkbenchStoreError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, safe_service_error(exc))
        except BaseException as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, safe_service_error(exc))

    def _read_body(self) -> dict[str, Any]:
        media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise WorkbenchError("the local request must use JSON")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError as exc:
            raise WorkbenchError("the local request size is invalid") from exc
        if not 0 <= length <= MAX_REQUEST_BYTES:
            raise WorkbenchError("the local request size is invalid")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError
            value = exact_json_loads(raw.decode("utf-8"))
        except (
            OSError, UnicodeError, json.JSONDecodeError, ValueError
        ) as exc:
            raise WorkbenchError("the local request JSON is invalid") from exc
        if not isinstance(value, dict):
            raise WorkbenchError("the local request JSON is invalid")
        return value

    @staticmethod
    def _require_keys(value: Mapping[str, Any], keys: set[str]) -> None:
        if set(value) != keys:
            raise WorkbenchError("the local request fields are invalid")

    def _post_save(self, value: Mapping[str, Any], *, qc: bool) -> None:
        keys = {"review_id", "expected_revision", "annotation", "submit"}
        self._require_keys(value, keys)
        if (
            not isinstance(value["review_id"], str)
            or not value["review_id"]
            or isinstance(value["expected_revision"], bool)
            or not isinstance(value["expected_revision"], int)
            or value["expected_revision"] < 0
            or not isinstance(value["annotation"], Mapping)
            or not isinstance(value["submit"], bool)
        ):
            raise WorkbenchError("the local request fields are invalid")
        if qc:
            if value["submit"] is not True:
                raise WorkbenchError("the local request fields are invalid")
            result = self.server.service.save_qc(
                row_id=value["review_id"],
                expected_revision=value["expected_revision"],
                annotation=value["annotation"],
            )
        else:
            result = self.server.service.save(
                row_id=value["review_id"],
                expected_revision=value["expected_revision"],
                annotation=value["annotation"],
                submit=value["submit"],
            )
        self._json(HTTPStatus.OK, result)

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_host():
            return
        target = urlsplit(self.path)
        if target.query or target.path not in {
            "/api/save",
            "/api/qc/start",
            "/api/qc/submit",
            "/api/proposals/reveal",
            "/api/session/close",
        }:
            self._error(HTTPStatus.NOT_FOUND, "The local route was not found.")
            return
        if not self._mutation_valid():
            self._error(HTTPStatus.FORBIDDEN, "The local workbench token is invalid.")
            return
        try:
            value = self._read_body()
            if target.path == "/api/save":
                self._post_save(value, qc=False)
            elif target.path == "/api/qc/start":
                self._require_keys(value, set())
                self._json(HTTPStatus.OK, self.server.service.start_qc())
            elif target.path == "/api/qc/submit":
                self._post_save(value, qc=True)
            elif target.path == "/api/proposals/reveal":
                self._require_keys(value, {"review_id"})
                if not isinstance(value["review_id"], str) or not value["review_id"]:
                    raise WorkbenchError("the local request fields are invalid")
                self._json(
                    HTTPStatus.OK,
                    {"proposals": self.server.service.reveal_proposals(value["review_id"])},
                )
            else:
                self._require_keys(value, set())
                self._json(HTTPStatus.OK, {"closed": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
        except StaleRevisionError as exc:
            self._error(HTTPStatus.CONFLICT, safe_service_error(exc))
        except AnnotationValidationError as exc:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                str(exc),
                problems=[item.as_dict() for item in exc.problems],
            )
        except (WorkbenchError, WorkbenchStoreError, TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, safe_service_error(exc))
        except BaseException as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, safe_service_error(exc))
