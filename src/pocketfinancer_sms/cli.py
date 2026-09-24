"""Aggregate-safe command line entry points."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from .corpus import build_private_corpus
from .evaluation import EvaluationInterrupted, evaluate_extractor
from .provenance import PrivateArtifactError, file_sha256, require_private_output
from .workbench.native_import import AesGcmBundleDecryptor, NativeTraceImporter
from .workbench.secure_store import (
    KeyringKeyProvider,
    SecureWorkbenchStore,
    migrate_plaintext_store,
)
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
    export = subparsers.add_parser(
        "export-workbench", help="export explicitly selected revisions as an encrypted bundle"
    )
    export.add_argument(
        "--selection-manifest",
        type=Path,
        required=True,
        help="private JSON list of exact source, reviewer, revision, and revision hash",
    )
    export.add_argument(
        "--consent-local-encrypted-export",
        action="store_true",
        help="confirm this explicit local encrypted export",
    )
    subparsers.add_parser(
        "migrate-secure-workbench",
        help="create and verify an encrypted replacement for the legacy workbench",
    )
    native_import = subparsers.add_parser(
        "import-native-trace",
        help="explicitly import one encrypted Android or iOS trace bundle",
    )
    native_import.add_argument("--bundle", type=Path, required=True)
    native_import.add_argument("--key-id", default="native-trace-import-v1")
    evaluation = subparsers.add_parser(
        "evaluate-extractor",
        help="evaluate the direct extractor with a local GGUF",
    )
    evaluation.add_argument("--gguf", type=Path, required=True)
    evaluation.add_argument(
        "--suite",
        choices=("synthetic", "grandfathered", "private-canonical"),
        required=True,
    )
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--account-catalog", type=Path)
    evaluation.add_argument("--primary-currency", default="INR")
    evaluation.add_argument("--profile", action="append", dest="profiles")
    evaluation.add_argument("--n-ctx", type=int, default=4096)
    evaluation.add_argument("--n-gpu-layers", type=int, default=-1)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--max-tokens", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-corpus":
            summary = build_private_corpus(args.repo_root, args.config)
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.command == "evaluate-extractor":
            try:
                summary = evaluate_extractor(
                    args.repo_root,
                    gguf=args.gguf,
                    suite=args.suite,
                    output_dir=args.output_dir,
                    account_catalog=args.account_catalog,
                    primary_currency=args.primary_currency,
                    enabled_profile_ids=tuple(args.profiles or ("core-en", "india")),
                    n_ctx=args.n_ctx,
                    n_gpu_layers=args.n_gpu_layers,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                )
            except EvaluationInterrupted as exc:
                print(
                    json.dumps(
                        {
                            "status": "interrupted",
                            "completed_rows": exc.completed_rows,
                            "total_rows": exc.total_rows,
                        },
                        sort_keys=True,
                    )
                )
                return 130
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.command == "migrate-secure-workbench":
            private_root = _private_root(args.repo_root)
            workbench_root = private_root / "workbench"
            source = require_private_output(
                args.repo_root.resolve(), workbench_root / "workbench.sqlite3"
            )
            destination = require_private_output(
                args.repo_root.resolve(), workbench_root / "workbench-v2.sqlite3"
            )
            provider = KeyringKeyProvider()
            result = migrate_plaintext_store(
                source,
                destination,
                key_provider=provider,
            )
            secure_store = SecureWorkbenchStore(
                destination,
                key_provider=provider,
            )
            recovery = secure_store.create_backup(workbench_root / "encrypted-recovery")
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "encrypted": True,
                        "source_count": result["source_count"],
                        "destination_count": result["destination_count"],
                        "recovery_verified": recovery["encrypted"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        store, private_root = _current_workbench(args.repo_root)
        _require_secure_store(store)
        if args.command == "init-workbench":
            count = _import_current_manifest(store, private_root)
            print(json.dumps({"status": "ready", "row_count": count}, sort_keys=True))
            return 0
        if args.command == "serve-workbench":
            _import_current_manifest(store, private_root)
            importer = NativeTraceImporter(
                store,
                expected_release_hash=_release_manifest_sha256(args.repo_root),
            )
            server = WorkbenchWebServer(
                WorkbenchService(store, native_trace_importer=importer),
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
        if args.command == "import-native-trace":
            provider = KeyringKeyProvider()
            decryptor = AesGcmBundleDecryptor(provider.get_existing_key(args.key_id))
            importer = NativeTraceImporter(
                store,
                expected_release_hash=_release_manifest_sha256(args.repo_root),
            )
            result = importer.import_bundle(
                args.bundle.resolve(),
                decryptor=decryptor,
                imported_at_epoch_ms=time.time_ns() // 1_000_000,
            )
            print(
                json.dumps(
                    {
                        "status": "imported",
                        "record_count": result["record_count"],
                        "replayed": result["replayed"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "export-workbench":
            selection_path = args.selection_manifest.resolve()
            if not selection_path.is_relative_to(private_root.resolve()):
                raise PrivateArtifactError("export selection must stay under PRIVATE_DATA")
            try:
                selected_revisions = json.loads(selection_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PrivateArtifactError("export selection manifest is unreadable") from exc
            result = store.export_labels(
                private_root / "workbench" / "exports",
                explicit_consent=args.consent_local_encrypted_export,
                selected_revisions=selected_revisions,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
    except PrivateArtifactError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, sort_keys=True))
        return 2
    return 2


def _current_workbench(repo_root: Path) -> tuple[WorkbenchStore, Path]:
    repo_root = repo_root.resolve()
    private_root = _private_root(repo_root)
    secure_database = require_private_output(
        repo_root, private_root / "workbench" / "workbench-v2.sqlite3"
    )
    if secure_database.exists():
        return (
            SecureWorkbenchStore(
                secure_database,
                key_provider=KeyringKeyProvider(),
            ),
            private_root,
        )
    legacy_database = require_private_output(
        repo_root, private_root / "workbench" / "workbench.sqlite3"
    )
    if legacy_database.exists():
        raise PrivateArtifactError(
            "legacy workbench requires explicit migrate-secure-workbench before use"
        )
    return (
        SecureWorkbenchStore(
            secure_database,
            key_provider=KeyringKeyProvider(),
        ),
        private_root,
    )


def _require_secure_store(store: WorkbenchStore) -> None:
    if not isinstance(store, SecureWorkbenchStore):
        raise PrivateArtifactError("workbench command requires encrypted SQLCipher storage")


def _private_root(repo_root: Path) -> Path:
    resolved = repo_root.resolve()
    return require_private_output(
        resolved, resolved / "PRIVATE_DATA" / "sms_processing"
    )


def _release_manifest_sha256(repo_root: Path) -> str:
    manifest = (
        repo_root.resolve()
        / "configs"
        / "sms_processing"
        / "contracts"
        / "releases"
        / "native-integration-v2.json"
    )
    try:
        return file_sha256(manifest)
    except OSError as exc:
        raise PrivateArtifactError("native integration release manifest is unavailable") from exc


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
