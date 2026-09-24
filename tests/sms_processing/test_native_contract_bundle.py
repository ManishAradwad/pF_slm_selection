"""Focused export and verification tests for the frozen native v3 app bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.package_native_contract_v3 import (
    BundleIntegrityError,
    INTEGRITY_REASON,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    check_bundle,
    export_bundle,
    load_frozen_assets,
    main,
    _safe_relative_path,
)


ROOT = Path(__file__).resolve().parents[2]


def _target(root: Path, path: object) -> Path:
    return root.joinpath(*str(path).split("/"))


@pytest.mark.parametrize(
    "path",
    [
        "../PRIVATE_DATA/secret",
        "/absolute/path",
        "configs//asset.json",
        "./configs/asset.json",
        "configs\\asset.json",
    ],
)
def test_source_allowlist_rejects_paths_that_could_escape_or_change_shape(path: str) -> None:
    with pytest.raises(BundleIntegrityError, match="unsafe artifact path"):
        _safe_relative_path(path)


def test_frozen_allowlist_includes_exact_manifest_and_every_referenced_asset() -> None:
    assets = load_frozen_assets()
    manifest = json.loads(_target(ROOT, MANIFEST_PATH).read_text(encoding="utf-8"))

    assert assets[0].path == MANIFEST_PATH
    assert assets[0].sha256 == MANIFEST_SHA256
    assert [asset.path.as_posix() for asset in assets[1:]] == [
        item["path"] for item in manifest["artifacts"]
    ]
    assert [asset.sha256 for asset in assets[1:]] == [
        item["sha256"] for item in manifest["artifacts"]
    ]


def test_export_preserves_paths_and_exact_bytes_then_is_idempotent(tmp_path: Path) -> None:
    historical = (
        tmp_path
        / "configs"
        / "sms_processing"
        / "contracts"
        / "releases"
        / "native-integration-v2.json"
    )
    historical.parent.mkdir(parents=True)
    historical.write_bytes(b"existing historical app asset")
    first = export_bundle(tmp_path)
    assets = load_frozen_assets()

    assert first == {
        "artifact_count": len(assets) - 1,
        "copied_files": len(assets),
        "existing_files": 0,
        "file_count": len(assets),
        "manifest_path": MANIFEST_PATH.as_posix(),
        "manifest_sha256": MANIFEST_SHA256,
        "release_id": "native-integration-v3",
        "status": "exported",
    }
    for asset in assets:
        assert _target(tmp_path, asset.path).read_bytes() == asset.content
    assert historical.read_bytes() == b"existing historical app asset"

    second = export_bundle(tmp_path)
    assert second["copied_files"] == 0
    assert second["existing_files"] == len(assets)
    assert check_bundle(tmp_path)["status"] == "verified"


def test_export_preflights_all_collisions_without_partial_copy(tmp_path: Path) -> None:
    assets = load_frozen_assets()
    conflicting = _target(tmp_path, assets[-1].path)
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"altered")

    with pytest.raises(BundleIntegrityError, match="refusing to overwrite"):
        export_bundle(tmp_path)

    assert conflicting.read_bytes() == b"altered"
    assert not _target(tmp_path, MANIFEST_PATH).exists()


@pytest.mark.parametrize("failure", ["missing", "altered"])
def test_check_fails_closed_for_missing_or_altered_assets(
    tmp_path: Path, failure: str, capsys: pytest.CaptureFixture[str]
) -> None:
    export_bundle(tmp_path)
    asset = load_frozen_assets()[-1]
    packaged = _target(tmp_path, asset.path)
    if failure == "missing":
        packaged.unlink()
    else:
        packaged.write_bytes(b"altered")

    assert main(["check", "--bundle-root", str(tmp_path)]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["reason"] == INTEGRITY_REASON
    assert asset.path.as_posix() in payload["summary"]
