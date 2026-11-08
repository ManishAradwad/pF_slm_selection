"""Isolated, supplied-response pipeline mechanics runner (not a quality evaluator).

Uses the existing advisory analyzer, strict extractor/grounding/exact money and
successor routing. No model SDK or private corpus is opened. SQLite here is for
invented evaluation state only, not an application storage/retention policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .account_resolution import AccountCatalogEntry, normalize_account_alias, transaction_fingerprint
from .admission import admission_action
from .analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from .currency import CurrencyContext
from .prefilter import POLICY_ID, prefilter
from .processing_v3 import (
    SuccessorExtractionCoordinator,
    processing_result_payload_v3,
)
from .types import TimestampProvenance

RUN_CONTRACT = "synthetic-stateful-pipeline/1"


def _json(value: Any) -> str:
    # ASCII-safe JSON preserves even malformed Unicode responses reversibly.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _host_source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "stateful_evaluation.py", "admission.py", "prefilter.py", "processing_v3.py",
            "extractor.py", "analyzer.py", "account_resolution.py", "currency.py",
            "structural_text.py", "profiles.py", "types.py", "feedback.py", "corpus/grouping.py",
        )
    }


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InventedMessage:
    body: str
    sender: str
    received_at_epoch_ms: int
    connector: str | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.body or len(self.body.encode("utf-8")) > 16_384:
            raise ValueError("invalid invented body")
        if type(self.received_at_epoch_ms) is not int or self.received_at_epoch_ms < 0:
            raise ValueError("invalid receipt time")
        if (self.connector is None) != (self.source_id is None):
            raise ValueError("source identity requires both connector and ID")
        if self.connector == "" or self.source_id == "":
            raise ValueError("source identity must not be empty")

    @property
    def reliable_key(self) -> str | None:
        if self.source_id is None:
            return None
        return _hash([self.connector, self.source_id])


@dataclass(frozen=True)
class RunReceipt:
    operation_id: str
    action: str
    route: str
    model_invoked: bool = False


class SuppliedRuntime:
    """A counted fake runtime; responses are supplied, never inferred/repaired."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any], _cancellation: Any) -> str:
        self.calls += 1
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("no supplied response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class SyntheticPipelineRun:
    """Durable admission -> prefilter -> snapshot -> existing host flow -> state.

    A fresh run refuses an existing file. Resume refuses any changed run config;
    ready work is bound to its original settings. Interrupted runtime work needs
    explicit recovery and owner retry, never a blind second call on replay.
    Separate instances serialize admission/settlement through SQLite. Do not use
    this invented-data store for private evaluation or as native storage parity.
    """

    def __init__(
        self,
        path: Path,
        *,
        primary_currency: str = "INR",
        enabled_profiles: tuple[str, ...] = ("core-en", "india"),
        resume: bool = False,
    ) -> None:
        self.settings = {
            "contract": RUN_CONTRACT,
            "primary_currency": primary_currency,
            "enabled_profiles": list(enabled_profiles),
            "prefilter": POLICY_ID,
            "host_source_hashes": _host_source_hashes(),
        }
        # Validate configuration before creating a run file.
        context = CurrencyContext(primary_currency, enabled_profiles)
        self.settings["primary_currency"] = context.primary_currency
        self.settings["analyzer_configuration_hash"] = context.config_hash
        if path.is_symlink() or (resume and not path.is_file()):
            raise ValueError("resume requires an existing regular run file")
        if not resume:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA busy_timeout = 5000")
        try:
            if not resume:
                self.db.executescript("""
                    CREATE TABLE run (configuration TEXT NOT NULL);
                    CREATE TABLE operations (
                        id TEXT PRIMARY KEY, source_key TEXT, evidence TEXT NOT NULL,
                        settings TEXT NOT NULL, parent_id TEXT REFERENCES operations(id),
                        state TEXT NOT NULL, route TEXT NOT NULL,
                        payload TEXT, fingerprint TEXT
                    );
                    CREATE UNIQUE INDEX ordinary_source ON operations(source_key)
                        WHERE parent_id IS NULL AND source_key IS NOT NULL;
                    CREATE TABLE admissions (
                        operation_id TEXT NOT NULL REFERENCES operations(id),
                        action TEXT NOT NULL, route TEXT NOT NULL
                    );
                    CREATE TABLE attempts (
                        operation_id TEXT PRIMARY KEY REFERENCES operations(id),
                        raw_output TEXT
                    );
                    CREATE TABLE accounts (
                        id TEXT PRIMARY KEY, kind TEXT NOT NULL, aliases TEXT NOT NULL
                    );
                    CREATE TABLE ledger (
                        source_key TEXT PRIMARY KEY, operation_id TEXT NOT NULL UNIQUE,
                        account_id TEXT NOT NULL REFERENCES accounts(id), payload TEXT NOT NULL
                    );
                """)
                self.db.execute("INSERT INTO run VALUES (?)", (_json(self.settings),))
            rows = self.db.execute("SELECT configuration FROM run").fetchall()
            if len(rows) != 1 or rows[0][0] != _json(self.settings):
                raise ValueError("run configuration mismatch; create a fresh isolated run")
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def admit(self, message: InventedMessage, *, retry_of: str | None = None) -> RunReceipt:
        evidence = _json(asdict(message))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = None
            if retry_of is not None:
                existing = self._operation(retry_of)
                # Retry must target the latest operation, not a stale parent.
                if self.db.execute(
                    "SELECT 1 FROM operations WHERE parent_id = ?", (retry_of,)
                ).fetchone():
                    raise ValueError("retry must target the latest operation")
            elif message.reliable_key is not None:
                existing = self.db.execute(
                    "SELECT * FROM operations WHERE source_key = ? ORDER BY rowid DESC LIMIT 1",
                    (message.reliable_key,),
                ).fetchone()
            action = admission_action(
                reliable_identity=message.reliable_key is not None,
                existing_state=existing["state"] if existing else None,
                same_evidence=existing["evidence"] == evidence if existing else True,
                owner_retry=retry_of is not None,
            )
            if action == "conflict":
                raise ValueError("source identity conflicts with immutable evidence")
            if action in {"reuse", "resume", "in_flight"}:
                assert existing is not None
                receipt = RunReceipt(existing["id"], action, existing["route"])
            else:
                operation_id = str(uuid.uuid4())
                decision = prefilter(message.body)
                excluded = decision.decision == "exclude"
                self.db.execute(
                    "INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        operation_id, message.reliable_key, evidence, _json(self.settings),
                        retry_of, "settled" if excluded else "ready",
                        "excluded" if excluded else "pending",
                        _json({"prefilter_reason": decision.reason}) if excluded else None,
                    ),
                )
                receipt = RunReceipt(operation_id, action, "excluded" if excluded else "pending")
            self.db.execute("INSERT INTO admissions VALUES (?, ?, ?)",
                            (receipt.operation_id, receipt.action, receipt.route))
            self.db.execute("COMMIT")
            return receipt
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def execute(self, operation_id: str, runtime: SuppliedRuntime) -> RunReceipt:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self._operation(operation_id)
            if row["state"] != "ready":
                self.db.execute("COMMIT")
                return RunReceipt(operation_id, "in_flight" if row["state"] == "in_flight"
                                  else "reuse", row["route"])
            if self.db.execute("SELECT 1 FROM operations WHERE parent_id = ?", (operation_id,)).fetchone():
                self.db.execute("COMMIT")
                return RunReceipt(operation_id, "reuse", "superseded")
            self.db.execute("UPDATE operations SET state = 'in_flight' WHERE id = ?", (operation_id,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        settings = json.loads(row["settings"])
        message = InventedMessage(**json.loads(row["evidence"]))
        profiles = tuple(settings["enabled_profiles"])
        catalog = tuple(
            AccountCatalogEntry(item["id"], item["kind"], tuple(json.loads(item["aliases"])))
            for item in self.db.execute("SELECT * FROM accounts ORDER BY id")
        )
        fingerprints = frozenset(
            item[0] for item in self.db.execute(
                "SELECT fingerprint FROM operations WHERE fingerprint IS NOT NULL AND id != ?",
                (operation_id,),
            )
        )
        coordinator = SuccessorExtractionCoordinator(
            DeterministicSmsAnalyzer(
                CurrencyContext(settings["primary_currency"], profiles),
                analysis_contract=ANALYSIS_CONTRACT_V2,
            ),
            runtime,
            primary_currency=settings["primary_currency"],
            enabled_profile_ids=profiles,
            account_catalog=catalog,
        )
        def invoke(request: dict[str, Any], cancellation: Any) -> str:
            # A durable attempt record precedes even the fake model call. A lost
            # response is an interrupted attempt, never a free automatic replay.
            self.db.execute("INSERT INTO attempts VALUES (?, NULL)", (operation_id,))
            raw = runtime(request, cancellation)
            self.db.execute("UPDATE attempts SET raw_output = ? WHERE operation_id = ?",
                            (_json(raw), operation_id))
            return raw

        coordinator.model_invoke = invoke
        calls_before = runtime.calls
        try:
            outcome = coordinator.process(
                source=message.body, raw_sender=message.sender,
                operation_id=operation_id, operation_config_hash=_hash(settings),
                received_at_epoch_ms=message.received_at_epoch_ms,
                received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
                source_event_key=row["source_key"] or operation_id,
                idempotency_key=operation_id,
                persisted_source_event_keys=frozenset(
                    item[0] for item in self.db.execute("SELECT source_key FROM ledger")
                ),
                known_transaction_fingerprints=fingerprints,
            )
            self.db.execute("BEGIN IMMEDIATE")
            # Explicit recovery may have fenced a late result from another instance.
            if self._operation(operation_id)["state"] != "in_flight":
                raise ValueError("runtime ownership was interrupted")
            if outcome.status == "eligible":
                assert outcome.duplicate_assessment is not None
                fingerprint = outcome.duplicate_assessment.transaction_fingerprint
                collision = self.db.execute(
                    "SELECT 1 FROM operations WHERE fingerprint = ? AND id != ?",
                    (fingerprint, operation_id),
                ).fetchone()
                persisted_sources = frozenset(
                    item[0] for item in self.db.execute("SELECT source_key FROM ledger")
                )
                if collision or (row["source_key"] or operation_id) in persisted_sources:
                    # Reapply the existing host gate to the SAME supplied response
                    # under the write lock. This is validation, not another model
                    # attempt, repair, merge, or a new payment-duplicate policy.
                    raw = self.db.execute(
                        "SELECT raw_output FROM attempts WHERE operation_id = ?", (operation_id,)
                    ).fetchone()[0]
                    coordinator.model_invoke = lambda *_args: json.loads(raw)
                    outcome = coordinator.process(
                        source=message.body, raw_sender=message.sender,
                        operation_id=operation_id, operation_config_hash=_hash(settings),
                        received_at_epoch_ms=message.received_at_epoch_ms,
                        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
                        source_event_key=row["source_key"] or operation_id,
                        idempotency_key=operation_id,
                        persisted_source_event_keys=persisted_sources,
                        known_transaction_fingerprints=frozenset({fingerprint}) if collision
                        else frozenset(),
                    )
            payload = processing_result_payload_v3(outcome)
            payload["review"] = outcome.review_case.to_dict() if outcome.review_case else None
            route = {"eligible": "transactions", "not_posted": "no_transaction"}.get(
                outcome.status, "review"
            )
            fingerprint = (outcome.duplicate_assessment.transaction_fingerprint
                           if outcome.result and outcome.result.transaction
                           and outcome.duplicate_assessment else None)
            if route == "transactions":
                assert outcome.account_resolution and outcome.account_resolution.account_id
                self._save(row, outcome.account_resolution.account_id, payload)
            self.db.execute(
                "UPDATE operations SET state = 'settled', route = ?, payload = ?, fingerprint = ? "
                "WHERE id = ?", (route, _json(payload), fingerprint, operation_id),
            )
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            self.db.execute(
                "UPDATE operations SET state = 'interrupted', route = 'retryable' "
                "WHERE id = ? AND state = 'in_flight'", (operation_id,),
            )
            raise
        return RunReceipt(operation_id, "execute", route, runtime.calls > calls_before)

    def recover_interrupted(self) -> int:
        """Explicit run-owner recovery only, after the old runner has stopped.

        Not called on ordinary resume: opening a second instance must not steal
        active work. A recovered operation never automatically calls a model.
        """
        return self.db.execute(
            "UPDATE operations SET state = 'interrupted', route = 'retryable' "
            "WHERE state = 'in_flight'"
        ).rowcount

    def confirm_account(self, operation_id: str, account_id: str, *, kind: str = "bank_account") -> None:
        """Simulated explicit owner action, not a model success or training label.

        Only a strictly validated posted result held for unresolved account may
        use this narrow mechanics helper. Other Review corrections are not
        implemented here. Account alias + ledger + outcome settle atomically.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self._operation(operation_id)
            if self.db.execute("SELECT 1 FROM operations WHERE parent_id = ?", (operation_id,)).fetchone():
                raise ValueError("confirmation must target the latest operation")
            payload = json.loads(row["payload"] or "null")
            if row["route"] != "review" or not payload or payload["recognition_decision"] != "posted":
                raise ValueError("confirmation requires a validated posted Review")
            if payload["reason_codes"] != ["account_resolution_unresolved"]:
                raise ValueError("other Review reasons require separate resolution")
            semantic = payload["semantic_result"]
            fingerprint = transaction_fingerprint(
                minor_units=semantic["money"]["minor_units"],
                currency=semantic["money"]["currency"], direction=semantic["direction"],
                account_id=account_id,
                received_at_epoch_ms=payload["receipt_timestamp"]["epoch_ms"],
            )
            if self.db.execute(
                "SELECT 1 FROM operations WHERE fingerprint = ? AND id != ?",
                (fingerprint, operation_id),
            ).fetchone():
                raise ValueError("possible duplicate requires separate owner resolution")
            reference = semantic["account_reference"]
            entry = AccountCatalogEntry(account_id, kind, (reference,))
            existing = self.db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
            aliases = list(json.loads(existing["aliases"])) if existing else []
            if existing and existing["kind"] != kind:
                raise ValueError("account type conflict")
            if reference not in aliases:
                aliases.append(reference)
            # Refuse to introduce alias ambiguity or silently reassign ownership.
            for other in self.db.execute("SELECT * FROM accounts WHERE id != ?", (account_id,)):
                if any(normalize_account_alias(alias) == normalize_account_alias(reference)
                       for alias in json.loads(other["aliases"])):
                    raise ValueError("account alias already belongs to another account")
            self.db.execute(
                "INSERT INTO accounts VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET aliases=excluded.aliases",
                (entry.account_id, entry.account_type, _json(aliases)),
            )
            self._save(row, account_id, payload)
            self.db.execute(
                "UPDATE operations SET route = 'transactions', fingerprint = ? WHERE id = ?",
                (fingerprint, operation_id),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def mechanics_summary(self) -> dict[str, int]:
        """All admitted inputs including exclusions/replays; not quality scores."""
        return {
            "inputs": self.db.execute("SELECT COUNT(*) FROM admissions").fetchone()[0],
            "operations": self.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0],
            "model_calls": self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            "excluded_inputs": self.db.execute(
                "SELECT COUNT(*) FROM admissions WHERE route = 'excluded'"
            ).fetchone()[0],
            "reused_inputs": self.db.execute(
                "SELECT COUNT(*) FROM admissions WHERE action IN ('reuse', 'in_flight')"
            ).fetchone()[0],
            "transactions": self.db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0],
            "review_operations": self.db.execute(
                "SELECT COUNT(*) FROM operations WHERE route = 'review'"
            ).fetchone()[0],
        }

    def _save(self, row: sqlite3.Row, account_id: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO ledger VALUES (?, ?, ?, ?)",
            (row["source_key"] or row["id"], row["id"], account_id, _json(payload)),
        )

    def _operation(self, operation_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM operations WHERE id = ?", (operation_id,)).fetchone()
        if row is None:
            raise ValueError("unknown operation")
        return row
