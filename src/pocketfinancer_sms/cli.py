"""Aggregate-safe command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .corpus import build_private_corpus
from .provenance import PrivateArtifactError, require_private_output
from .workbench.service import WorkbenchService
from .workbench.store import WorkbenchStore
from .workbench.web import WorkbenchWebServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PocketFinancer local SMS processing tools")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("build-corpus", help="build the ignored canonical corpus")
    corpus.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sms_processing/archive-india-inr.json"),
    )
    subparsers.add_parser("init-workbench", help="import the current canonical manifest")
    serve = subparsers.add_parser("serve-workbench", help="serve the loopback-only workbench")
    serve.add_argument("--port", type=int, default=8765)
    subparsers.add_parser("backup-workbench", help="create and verify a local SQLite backup")
    subparsers.add_parser("export-workbench", help="export submitted canonical labels")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-corpus":
            summary = build_private_corpus(args.repo_root, args.config)
            print(json.dumps(summary, sort_keys=True))
            return 0
        store, private_root = _current_workbench(args.repo_root)
        if args.command == "init-workbench":
            count = _import_current_manifest(store, private_root)
            print(json.dumps({"status": "ready", "row_count": count}, sort_keys=True))
            return 0
        if args.command == "serve-workbench":
            _import_current_manifest(store, private_root)
            server = WorkbenchWebServer(
                WorkbenchService(store),
                port=args.port,
                backup_root=private_root / "workbench" / "backups",
                export_root=private_root / "workbench" / "exports",
            )
            print(json.dumps({"status": "serving", "url": server.url}, sort_keys=True))
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.close()
            return 0
        if args.command == "backup-workbench":
            result = store.create_backup(private_root / "workbench" / "backups")
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "export-workbench":
            result = store.export_labels(private_root / "workbench" / "exports")
            print(json.dumps(result, sort_keys=True))
            return 0
    except PrivateArtifactError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, sort_keys=True))
        return 2
    return 2


def _current_workbench(repo_root: Path) -> tuple[WorkbenchStore, Path]:
    repo_root = repo_root.resolve()
    private_root = require_private_output(
        repo_root, repo_root / "PRIVATE_DATA" / "sms_processing"
    )
    database = require_private_output(
        repo_root, private_root / "workbench" / "workbench.sqlite3"
    )
    return WorkbenchStore(database), private_root


def _import_current_manifest(store: WorkbenchStore, private_root: Path) -> int:
    try:
        current = json.loads((private_root / "CURRENT.json").read_text(encoding="utf-8"))
        run_id = current["run_id"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PrivateArtifactError("current canonical corpus pointer is invalid") from exc
    if not isinstance(run_id, str) or not run_id:
        raise PrivateArtifactError("current canonical corpus pointer is invalid")
    manifest = private_root / "runs" / run_id / "canonical_manifest.jsonl"
    return store.import_manifest(manifest, corpus_run_id=run_id)
