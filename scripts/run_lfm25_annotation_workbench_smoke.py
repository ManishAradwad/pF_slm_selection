#!/usr/bin/env python3
"""Launch the real annotation UI over invented, ephemeral smoke-test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import threading


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lfm25.annotation_service import AnnotationService  # noqa: E402
from lfm25.annotation_sources import (  # noqa: E402
    source_prefill_binding,
    source_prefill_for_sms,
)
from lfm25.annotation_store import WorkbenchStore, WorkbenchStoreError  # noqa: E402
from lfm25.annotation_web import WorkbenchHTTPServer  # noqa: E402
from lfm25.annotation_workbench import (  # noqa: E402
    SOURCE_PREFILL_OFF,
    SOURCE_PREFILL_POLICIES,
    TRAINING_MODE,
    WORKBENCH_CONTRACT,
    WORKBENCH_SCHEMA_VERSION,
    WorkbenchError,
    WorkbenchSourceRow,
    WorkspaceDefinition,
    exact_json_dumps,
)


SYNTHETIC_REVIEWER = "synthetic-smoke-reviewer"
SYNTHETIC_NOTICE = "SYNTHETIC DEMO ONLY - NO PRIVATE DATA"


def _port(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the real local annotation UI with three invented rows and "
            "ephemeral state. No private-data loader or export path is used."
        )
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=0,
        help="Loopback port; 0 (the default) asks the OS for an unused port.",
    )
    parser.add_argument(
        "--source-prefill",
        choices=SOURCE_PREFILL_POLICIES,
        default=SOURCE_PREFILL_OFF,
        help=(
            "Optional deterministic source-only prefill; off by default. "
            "Use unambiguous only to exercise the assisted methodology."
        ),
    )
    return parser


def synthetic_definition(
    *, source_prefill: str = SOURCE_PREFILL_OFF
) -> WorkspaceDefinition:
    fixtures = (
        (
            "synthetic-smoke-transaction-debit",
            "SYNTH-DEMO-BANK",
            (
                f"[{SYNTHETIC_NOTICE}] Acme Demo Bank: INR 42.50 was debited "
                "from A/c XX1234 at Paper Kite Cafe."
            ),
            ("candidate_coverage_miss",),
        ),
        (
            "synthetic-smoke-transaction-credit",
            "SYNTH-DEMO-PAY",
            (
                f"[{SYNTHETIC_NOTICE}] INR 125.00 was credited to A/c "
                "XX9876 from Sample Sender."
            ),
            ("multiple_entities",),
        ),
        (
            "synthetic-smoke-null",
            "SYNTH-DEMO-SECURITY",
            (
                f"[{SYNTHETIC_NOTICE}] Your invented verification code is 654321. "
                "Do not share it."
            ),
            ("otp_or_security",),
        ),
    )
    rows = tuple(
        WorkbenchSourceRow(
            row_id=row_id,
            position=position,
            sender=sender,
            sms=sms,
            source_json=exact_json_dumps(
                {
                    "record_hash": row_id,
                    "split": "train",
                    "sender": sender,
                    "sms": sms,
                    "local_model_proposals": [],
                }
            ),
            split="train",
            queue_tags=queue_tags,
            source_prefill=source_prefill_for_sms(
                sender,
                sms,
                source_prefill=source_prefill,
            ),
        )
        for position, (row_id, sender, sms, queue_tags) in enumerate(fixtures)
    )
    identity_digest = hashlib.sha256(
        ("\n".join(sorted(row.row_id for row in rows)) + "\n").encode("utf-8")
    ).hexdigest()
    return WorkspaceDefinition(
        mode=TRAINING_MODE,
        rows=rows,
        binding={
            "contract": WORKBENCH_CONTRACT,
            "mode": TRAINING_MODE,
            "fixture": "invented-synthetic-smoke-v1",
            "row_count": len(rows),
            "record_id_set_sha256": identity_digest,
            "ordering": "fixed_invented_fixture_order",
            **source_prefill_binding(
                SCRIPT_ROOT,
                rows,
                source_prefill=source_prefill,
            ),
        },
        metadata={
            "schema_version": WORKBENCH_SCHEMA_VERSION,
            "synthetic_only": True,
            "display_notice": SYNTHETIC_NOTICE,
            "raw_values_emitted_to_console": False,
        },
    )


def _serve(port: int, *, source_prefill: str) -> int:
    definition = synthetic_definition(source_prefill=source_prefill)
    with TemporaryDirectory(prefix="pf-annotation-smoke-") as temporary_directory:
        database = Path(temporary_directory) / "synthetic-smoke.sqlite3"
        store = WorkbenchStore(database, workspace_binding=definition.binding)
        server: WorkbenchHTTPServer | None = None
        try:
            service = AnnotationService(
                repo_root=SCRIPT_ROOT,
                definition=definition,
                store=store,
                reviewer=SYNTHETIC_REVIEWER,
                batch_size=definition.row_count,
            )
            server = WorkbenchHTTPServer(service, port=port)

            def request_shutdown(_signum: int, _frame: object) -> None:
                threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGINT, request_shutdown)
            signal.signal(signal.SIGTERM, request_shutdown)
            print(
                json.dumps(
                    {
                        "operation": "serve-synthetic-smoke",
                        "mode": definition.mode,
                        "notice": SYNTHETIC_NOTICE,
                        "rows": definition.row_count,
                        "url": server.local_url,
                        "bind": "127.0.0.1",
                        "synthetic_only": True,
                        "source_prefill": source_prefill,
                        "annotation_methodology": definition.binding.get(
                            "annotation_methodology", "human_verified"
                        ),
                        "persistent_state": False,
                        "cleanup_on_shutdown": True,
                        "raw_values_emitted": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            server.serve_forever(poll_interval=0.1)
            return 0
        finally:
            if server is not None:
                server.server_close()
            store.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return _serve(arguments.port, source_prefill=arguments.source_prefill)
    except (WorkbenchError, WorkbenchStoreError, OSError) as exc:
        parser.exit(2, f"synthetic annotation smoke failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
