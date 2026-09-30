"""Load private named source selections for the local annotation workbench."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..provenance import PrivateArtifactError, file_sha256
from .store import WorkbenchStore


PROTECTED_POOLS = frozenset({"protected_test", "later_time_holdout"})


@dataclass(frozen=True)
class SourceSet:
    set_id: str
    name: str
    source_ids: tuple[str, ...]

    def summary(self) -> dict[str, str | int]:
        return {"id": self.set_id, "name": self.name, "count": len(self.source_ids)}


class SourceSetCatalog:
    """Discover selections from local private artifacts, checking corpus provenance."""

    def __init__(self, store: WorkbenchStore, private_root: Path) -> None:
        self.store = store
        self.private_root = private_root.resolve()

    def list_sets(self) -> list[SourceSet]:
        evaluations_root = self.private_root / "evaluations"
        if not evaluations_root.is_dir():
            return []
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for directory in sorted(evaluations_root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict) and "set_id" in manifest:
                candidates.append((directory, manifest))
        if not candidates:
            return []
        run_id, canonical_sha256 = self._current_corpus()
        available: list[SourceSet] = []
        for directory, manifest in candidates:
            try:
                available.append(self._load(directory, manifest, run_id, canonical_sha256))
            except PrivateArtifactError:
                # A damaged optional set must not hide independently valid sets.
                continue
        return available

    def get_set(self, set_id: str) -> SourceSet:
        """Fail closed for a chosen set, including one omitted from the menu."""

        if (
            not set_id
            or not set_id.isascii()
            or not all(char.isalnum() or char == "-" for char in set_id)
        ):
            raise KeyError(set_id)
        directory = self.private_root / "evaluations" / set_id
        if not directory.is_dir():
            raise KeyError(set_id)
        if directory.is_symlink():
            raise PrivateArtifactError("private source set path is invalid")
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink():
            raise PrivateArtifactError("private source set manifest path is invalid")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateArtifactError("private source set manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise PrivateArtifactError("private source set manifest is invalid")
        run_id, canonical_sha256 = self._current_corpus()
        return self._load(directory, manifest, run_id, canonical_sha256)

    def _current_corpus(self) -> tuple[str, str]:
        run_id = self.store.corpus_run_id()
        try:
            current = json.loads(
                (self.private_root / "CURRENT.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateArtifactError("current canonical corpus pointer is invalid") from exc
        if (
            not isinstance(current, dict)
            or not isinstance(run_id, str)
            or not run_id
            or current.get("run_id") != run_id
        ):
            raise PrivateArtifactError("workbench corpus run is not current")
        runs_root = (self.private_root / "runs").resolve()
        canonical_path = (runs_root / run_id / "canonical_manifest.jsonl").resolve()
        if not canonical_path.is_relative_to(runs_root):
            raise PrivateArtifactError("current canonical corpus pointer is invalid")
        try:
            return run_id, file_sha256(canonical_path)
        except OSError as exc:
            raise PrivateArtifactError("current canonical manifest is unavailable") from exc

    def _load(
        self,
        directory: Path,
        manifest: dict[str, Any],
        run_id: str,
        canonical_sha256: str,
    ) -> SourceSet:
        set_id = manifest.get("set_id")
        if (
            not isinstance(set_id, str)
            or not set_id
            or set_id != directory.name
            or not all(char.isalnum() or char == "-" for char in set_id)
        ):
            raise PrivateArtifactError("private source set name is invalid")
        counts = manifest.get("counts")
        files = manifest.get("files")
        provenance = manifest.get("provenance")
        if (
            not isinstance(counts, dict)
            or not isinstance(files, dict)
            or not isinstance(provenance, dict)
        ):
            raise PrivateArtifactError("private source set manifest is incomplete")
        total = counts.get("total")
        file_metadata = files.get("selected_sources.jsonl")
        if (
            type(total) is not int
            or total < 1
            or not isinstance(file_metadata, dict)
            or type(file_metadata.get("records")) is not int
            or file_metadata["records"] != total
            or provenance.get("corpus_run_id") != run_id
            or provenance.get("canonical_manifest_sha256") != canonical_sha256
        ):
            raise PrivateArtifactError("private source set is stale or incomplete")
        selection_path = (directory / "selected_sources.jsonl").resolve()
        if not selection_path.is_relative_to(directory.resolve()):
            raise PrivateArtifactError("private source set path is invalid")
        try:
            selection_bytes = selection_path.read_bytes()
            selected_sha256 = hashlib.sha256(selection_bytes).hexdigest()
            if (
                file_metadata.get("sha256") != selected_sha256
                or provenance.get("selected_sources_sha256") != selected_sha256
            ):
                raise PrivateArtifactError("private source set selection hash differs")
            source_ids = self._read_source_ids(
                selection_bytes.decode("utf-8").splitlines()
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateArtifactError("private source set selection is unreadable") from exc
        if len(source_ids) != total or len(set(source_ids)) != total:
            raise PrivateArtifactError("private source set selection count is invalid")
        pool_counts = self.store.source_set_pool_counts(source_ids)
        if sum(pool_counts.values()) != total:
            raise PrivateArtifactError("private source set is missing from the current corpus")
        if PROTECTED_POOLS.intersection(pool_counts):
            raise PrivateArtifactError("private source set includes protected review rows")
        name = (
            "Cross-platform diversity v5"
            if set_id == "cross-platform-diversity-v5"
            else set_id.replace("-", " ").capitalize()
        )
        return SourceSet(set_id, name, source_ids)

    @staticmethod
    def _read_source_ids(lines: list[str]) -> tuple[str, ...]:
        source_ids: list[str] = []
        for line in lines:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise PrivateArtifactError("private source set selection is invalid")
            source_id = record.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise PrivateArtifactError("private source set selection is invalid")
            source_ids.append(source_id)
        return tuple(source_ids)
