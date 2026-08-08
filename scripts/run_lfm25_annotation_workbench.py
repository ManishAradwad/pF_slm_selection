#!/usr/bin/env python3
"""Launch the strictly local PocketFinancer annotation workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lfm25.annotation_service import AnnotationService  # noqa: E402
from lfm25.annotation_sources import (  # noqa: E402
    load_blinded_workspace,
    load_training_workspace,
    private_path,
)
from lfm25.annotation_store import (  # noqa: E402
    WorkbenchStore,
    WorkbenchStoreError,
    recover_backup,
    validate_backup,
)
from lfm25.annotation_web import WorkbenchHTTPServer  # noqa: E402
from lfm25.annotation_workbench import (  # noqa: E402
    DEFAULT_BLINDED_DB,
    DEFAULT_TRAINING_DB,
    DEFAULT_TRAINING_EXPORT,
    DEFAULT_TRAINING_REPORT,
    WorkbenchError,
)


def _path(value: str) -> Path:
    return Path(value)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return parsed


def _common(parser: argparse.ArgumentParser, *, default_db: Path) -> None:
    parser.add_argument("--repo-root", type=_path, default=SCRIPT_ROOT)
    parser.add_argument("--reviewer", required=True, help="Stable local reviewer identity.")
    parser.add_argument("--batch-size", type=_positive, default=50)
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--db", type=_path, default=default_db)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly local annotation workbench; binds only to 127.0.0.1."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    blinded = subparsers.add_parser("blinded", help="Open the frozen reviewer-blind test package.")
    _common(blinded, default_db=DEFAULT_BLINDED_DB)
    training = subparsers.add_parser("training", help="Open an explicit train/dev-only pool.")
    _common(training, default_db=DEFAULT_TRAINING_DB)
    training.add_argument("--pool", type=_path, required=True)
    export = subparsers.add_parser(
        "export-training", help="Export reviewed train/dev rows without overwriting."
    )
    export.add_argument("--repo-root", type=_path, default=SCRIPT_ROOT)
    export.add_argument("--reviewer", required=True, help="Stable local reviewer identity.")
    export.add_argument("--db", type=_path, default=DEFAULT_TRAINING_DB)
    export.add_argument("--pool", type=_path, required=True)
    export.add_argument("--output", type=_path, default=DEFAULT_TRAINING_EXPORT)
    export.add_argument("--report", type=_path, default=DEFAULT_TRAINING_REPORT)
    recovery = subparsers.add_parser("recover", help="Recover a workbench database backup.")
    recovery.add_argument("--repo-root", type=_path, default=SCRIPT_ROOT)
    recovery.add_argument("--db", type=_path, required=True)
    recovery.add_argument("--backup", type=_path, required=True)
    return parser


def _serve(arguments: argparse.Namespace) -> int:
    repo_root = arguments.repo_root.resolve()
    db_path = private_path(repo_root, arguments.db)
    if arguments.command == "blinded":
        definition = load_blinded_workspace(
            repo_root,
            include_initial_annotations=not db_path.exists(),
        )
    else:
        definition = load_training_workspace(
            repo_root,
            pool_file=arguments.pool,
            sealed_manifest=Path("PRIVATE_DATA/lfm25/split_manifest.jsonl"),
        )
    store = WorkbenchStore(db_path, workspace_binding=definition.binding)
    server: WorkbenchHTTPServer | None = None
    try:
        service = AnnotationService(
            repo_root=repo_root,
            definition=definition,
            store=store,
            reviewer=arguments.reviewer,
            batch_size=arguments.batch_size,
        )
        server = WorkbenchHTTPServer(service, port=arguments.port)

        def request_shutdown(_signum: int, _frame: object) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)
        print(
            json.dumps(
                {
                    "operation": "serve",
                    "mode": definition.mode,
                    "rows": definition.row_count,
                    "url": server.local_url,
                    "bind": "127.0.0.1",
                    "raw_values_emitted": False,
                },
                sort_keys=True,
            )
        )
        server.serve_forever(poll_interval=0.25)
        return 0
    finally:
        if server is not None:
            server.server_close()
        store.close()


def _export_training(arguments: argparse.Namespace) -> int:
    repo_root = arguments.repo_root.resolve()
    db_path = private_path(repo_root, arguments.db)
    definition = load_training_workspace(
        repo_root,
        pool_file=arguments.pool,
        sealed_manifest=Path("PRIVATE_DATA/lfm25/split_manifest.jsonl"),
    )
    store = WorkbenchStore(db_path, workspace_binding=definition.binding)
    try:
        service = AnnotationService(
            repo_root=repo_root,
            definition=definition,
            store=store,
            reviewer=arguments.reviewer,
        )
        report = service.export_training(
            output_file=arguments.output,
            report_file=arguments.report,
            force=False,
        )
    finally:
        store.close()
    print(json.dumps(report, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "export-training":
            return _export_training(arguments)
        if arguments.command == "recover":
            repo_root = arguments.repo_root.resolve()
            database = private_path(repo_root, arguments.db)
            backup = private_path(repo_root, arguments.backup)
            validation = validate_backup(database, backup)
            displaced = recover_backup(database, backup)
            print(
                json.dumps(
                    {
                        "operation": "recover",
                        "valid": True,
                        "schema_version": validation.schema_version,
                        "row_count": validation.row_count,
                        "event_count": validation.event_count,
                        "displaced_database_preserved": displaced is not None,
                        "raw_values_emitted": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        return _serve(arguments)
    except (WorkbenchError, WorkbenchStoreError, OSError) as exc:
        parser.exit(2, f"annotation workbench failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
