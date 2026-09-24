"""Fail-closed SQLCipher storage helpers for the local workbench.

The legacy :class:`WorkbenchStore` remains readable so an existing private run is
never rewritten implicitly. New encrypted stores opt in through
``SecureWorkbenchStore`` and refuse to open when SQLCipher or an OS-backed key
provider is unavailable.
"""

from __future__ import annotations

import importlib
import base64
import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Protocol

from ..provenance import (
    PrivateArtifactError,
    atomic_write_json,
    ensure_private_directory,
    file_sha256,
    fsync_directory,
)
from .store import SCHEMA_VERSION, WorkbenchStore


class WorkbenchKeyProvider(Protocol):
    """Returns a 32-byte database key without logging or persisting plaintext."""

    def get_or_create_key(self, key_id: str) -> bytes: ...


class KeyringKeyProvider:
    """OS credential-vault provider using the optional ``keyring`` package."""

    def __init__(self, service_name: str = "PocketFinancer SMS Workbench") -> None:
        self.service_name = service_name

    def get_or_create_key(self, key_id: str) -> bytes:
        try:
            keyring = importlib.import_module("keyring")
        except ImportError as exc:
            raise PrivateArtifactError(
                "OS-protected workbench keys require the optional keyring backend"
            ) from exc
        import base64
        import secrets

        encoded = keyring.get_password(self.service_name, key_id)
        if encoded is None:
            key = secrets.token_bytes(32)
            keyring.set_password(self.service_name, key_id, base64.b64encode(key).decode("ascii"))
            return key
        return _decode_key(encoded)

    def get_existing_key(self, key_id: str) -> bytes:
        """Read an explicitly provisioned import key without silently creating one."""

        try:
            keyring = importlib.import_module("keyring")
        except ImportError as exc:
            raise PrivateArtifactError(
                "OS-protected workbench keys require the optional keyring backend"
            ) from exc
        encoded = keyring.get_password(self.service_name, key_id)
        if encoded is None:
            raise PrivateArtifactError("OS-protected native trace import key is unavailable")
        return _decode_key(encoded)


def _sqlcipher_module():
    for name in ("sqlcipher3", "pysqlcipher3.dbapi2"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise PrivateArtifactError(
        "encrypted workbench storage requires a SQLCipher Python driver"
    )


class SecureWorkbenchStore(WorkbenchStore):
    """Workbench store whose SQLite pages and WAL are protected by SQLCipher."""

    def __init__(
        self,
        database_path: Path,
        *,
        key_provider: WorkbenchKeyProvider,
        key_id: str = "workbench-v2",
    ) -> None:
        self._key_provider = key_provider
        self._key_id = key_id
        self._cipher = _sqlcipher_module()
        super().__init__(database_path)

    def connect(self) -> sqlite3.Connection:
        key = self._key_provider.get_or_create_key(self._key_id)
        if len(key) != 32:
            raise PrivateArtifactError("encrypted workbench key must contain exactly 32 bytes")
        connection = self._cipher.connect(str(self.database_path), timeout=10)
        try:
            connection.execute(f"PRAGMA key = \"x'{key.hex()}'\"")
            version = connection.execute("PRAGMA cipher_version").fetchone()
            if version is None or not version[0]:
                raise PrivateArtifactError("database driver did not prove SQLCipher support")
            connection.row_factory = sqlite3.Row
            connection.create_function(
                "source_ref_sha256", 1,
                lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
                deterministic=True,
            )
            connection.execute("PRAGMA cipher_memory_security = ON")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 10000")
            return connection
        except Exception:
            connection.close()
            raise

    def create_backup(self, backup_root: Path) -> dict[str, object]:
        """Create a verified SQLCipher backup; never downgrade to plaintext SQLite."""

        self.verify_revision_chains()
        ensure_private_directory(backup_root)
        stamp = time.time_ns()
        temporary = backup_root / f".workbench-{stamp}.sqlite3.tmp"
        destination = backup_root / f"workbench-{stamp}.sqlite3"
        try:
            candidate = SecureWorkbenchStore(
                temporary,
                key_provider=self._key_provider,
                key_id=self._key_id,
            )
            with closing(self.connect()) as source, closing(candidate.connect()) as target:
                run_row = source.execute(
                    "SELECT value FROM meta WHERE key = 'corpus_run_id'"
                ).fetchone()
                if run_row is None:
                    raise PrivateArtifactError(
                        "workbench backup requires an imported corpus run"
                    )
                corpus_run_id = run_row[0]
                source.backup(target)
                target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                target.commit()
            candidate.verify_revision_chains()
            if not candidate.integrity_check():
                raise PrivateArtifactError(
                    "encrypted workbench backup failed integrity verification"
                )
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
            activated = SecureWorkbenchStore(
                destination,
                key_provider=self._key_provider,
                key_id=self._key_id,
            )
            activated.verify_revision_chains()
            if not activated.integrity_check():
                raise PrivateArtifactError(
                    "activated encrypted workbench backup failed integrity verification"
                )
        finally:
            temporary.unlink(missing_ok=True)
            Path(f"{temporary}-wal").unlink(missing_ok=True)
            Path(f"{temporary}-shm").unlink(missing_ok=True)
        digest = file_sha256(destination)
        manifest = {
            "backup": destination.name,
            "sha256": digest,
            "schema_version": SCHEMA_VERSION,
            "corpus_run_id": corpus_run_id,
            "encrypted": True,
        }
        atomic_write_json(destination.with_suffix(".manifest.json"), manifest)
        return {
            "backup": destination.name,
            "sha256": digest,
            "schema_version": SCHEMA_VERSION,
            "encrypted": True,
        }

    def export_labels(
        self,
        output_root: Path,
        *,
        explicit_consent: bool = False,
        selected_revisions: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Write only explicitly selected revisions to an encrypted export."""

        if not explicit_consent:
            raise PrivateArtifactError("encrypted workbench export requires explicit consent")
        self.verify_revision_chains()
        ensure_private_directory(output_root)
        with closing(self.connect()) as connection:
            run_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'corpus_run_id'"
            ).fetchone()
            if run_row is None:
                raise PrivateArtifactError("workbench export requires an imported corpus run")
        labels = self.selected_labels(selected_revisions)
        payload = {
            "contract": "pocketfinancer.workbench-export-payload/2",
            "corpus_run_id": run_row[0],
            "workbench_schema_version": SCHEMA_VERSION,
            "labels": labels,
        }
        plaintext = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        export_id = hashlib.sha256(plaintext).hexdigest()[:20]
        key = self._key_provider.get_or_create_key(self._key_id)
        if len(key) != 32:
            raise PrivateArtifactError("encrypted workbench key must contain exactly 32 bytes")
        try:
            aes_gcm = importlib.import_module(
                "cryptography.hazmat.primitives.ciphers.aead"
            ).AESGCM(key)
        except ImportError as exc:
            raise PrivateArtifactError(
                "encrypted workbench export requires the optional cryptography package"
            ) from exc
        nonce = secrets.token_bytes(12)
        associated_data = b"pocketfinancer.workbench-export/2"
        ciphertext = aes_gcm.encrypt(nonce, plaintext, associated_data)
        envelope = {
            "contract": "pocketfinancer.workbench-encrypted-export/2",
            "export_id": export_id,
            "label_count": len(labels),
            "workbench_schema_version": SCHEMA_VERSION,
            "explicit_consent": True,
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_protection": "os_protected",
                "nonce_base64": base64.b64encode(nonce).decode("ascii"),
                "associated_data": associated_data.decode("ascii"),
            },
            "payload_ciphertext_base64": base64.b64encode(ciphertext).decode("ascii"),
            "payload_ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        }
        destination = output_root / f"workbench-export-{export_id}.json"
        atomic_write_json(destination, envelope)
        return {
            "export_id": export_id,
            "label_count": len(labels),
            "encrypted": True,
        }


def migrate_plaintext_store(
    source_path: Path,
    destination_path: Path,
    *,
    key_provider: WorkbenchKeyProvider,
    key_id: str = "workbench-v2",
) -> dict[str, int]:
    """Copy and verify a legacy store without deleting or replacing its source."""

    if not source_path.is_file() or destination_path.exists():
        raise PrivateArtifactError("secure migration paths are not eligible")
    ensure_private_directory(destination_path.parent)
    source = WorkbenchStore(source_path)
    source.verify_revision_chains()
    key = key_provider.get_or_create_key(key_id)
    if len(key) != 32:
        raise PrivateArtifactError("encrypted workbench key must contain exactly 32 bytes")
    cipher = _sqlcipher_module()
    temporary = destination_path.with_name(
        f".{destination_path.name}.pending-{uuid.uuid4().hex}"
    )
    try:
        connection = cipher.connect(str(source_path), timeout=10)
        try:
            connection.execute("PRAGMA key = ''")
            version = connection.execute("PRAGMA cipher_version").fetchone()
            if version is None or not version[0]:
                raise PrivateArtifactError("database driver did not prove SQLCipher support")
            connection.execute(
                f"ATTACH DATABASE ? AS encrypted KEY \"x'{key.hex()}'\"",
                (str(temporary),),
            )
            connection.execute("SELECT sqlcipher_export('encrypted')")
            connection.execute("DETACH DATABASE encrypted")
        finally:
            connection.close()
        os.chmod(temporary, 0o600)
        pending = SecureWorkbenchStore(
            temporary,
            key_provider=key_provider,
            key_id=key_id,
        )
        pending.verify_revision_chains()
        if not pending.integrity_check():
            raise PrivateArtifactError(
                "encrypted workbench migration failed integrity verification"
            )
        with closing(source.connect()) as plain, closing(pending.connect()) as encrypted:
            source_counts = _table_counts(plain)
            destination_counts = _table_counts(encrypted)
            source_count = source_counts.get("corpus_rows", 0)
            destination_count = destination_counts.get("corpus_rows", 0)
            encrypted.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            encrypted.commit()
        if source_counts != destination_counts:
            raise PrivateArtifactError(
                "encrypted workbench migration did not preserve record counts"
            )
        os.replace(temporary, destination_path)
        fsync_directory(destination_path.parent)
        activated = SecureWorkbenchStore(
            destination_path,
            key_provider=key_provider,
            key_id=key_id,
        )
        activated.verify_revision_chains()
        if not activated.integrity_check():
            raise PrivateArtifactError(
                "activated encrypted workbench migration failed integrity verification"
            )
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)
    return {"source_count": source_count, "destination_count": destination_count}


def _decode_key(encoded: str) -> bytes:
    import base64

    try:
        key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise PrivateArtifactError("OS-protected workbench key is invalid") from exc
    if len(key) != 32:
        raise PrivateArtifactError("OS-protected workbench key has an invalid length")
    return key


def _table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    names = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    counts: dict[str, int] = {}
    for row in names:
        name = str(row[0])
        quoted = '"' + name.replace('"', '""') + '"'
        counts[name] = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    return counts
