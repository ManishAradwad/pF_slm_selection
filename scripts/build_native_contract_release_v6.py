#!/usr/bin/env python3
"""Build the additive native SMS release with operation-bound grammar mode."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pocketfinancer_sms.provenance import file_sha256  # noqa: E402

try:
    from scripts.build_native_contract_release_v5 import (  # noqa: E402
        build_manifest as build_v5_manifest,
    )
except ModuleNotFoundError:
    from build_native_contract_release_v5 import (  # type: ignore[no-redef]  # noqa: E402
        build_manifest as build_v5_manifest,
    )

V5_MANIFEST_PATH = (
    "configs/sms_processing/contracts/releases/native-integration-v5.json"
)
V5_MANIFEST_SHA256 = (
    "971ac758729916e8dfdd8ea165d4e8344b5d3ab6e4d190473c8e91e2541c5f6d"
)
NEW_ARTIFACTS = (
    (
        "configs/sms_processing/contracts/v6/processing-config.schema.json",
        "pocketfinancer.processing-config/6",
    ),
    (
        "configs/sms_processing/contracts/releases/v6/release-manifest.schema.json",
        "pocketfinancer.contract-release-manifest-schema/6",
    ),
)


def _read_json(path: str) -> Any:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def build_manifest() -> dict[str, Any]:
    predecessor = build_v5_manifest()
    if file_sha256(REPO_ROOT / V5_MANIFEST_PATH) != V5_MANIFEST_SHA256:
        raise ValueError("frozen native-integration-v5 manifest changed")
    if _read_json(V5_MANIFEST_PATH) != predecessor:
        raise ValueError("native-integration-v5 no longer matches its deterministic rebuild")

    for path, contract in NEW_ARTIFACTS:
        schema = _read_json(path)
        if schema.get("$id") != contract:
            raise ValueError(f"{path} declares the wrong contract")
        jsonschema.Draft202012Validator.check_schema(schema)

    manifest = copy.deepcopy(predecessor)
    manifest["release_id"] = "native-integration-v6"
    manifest["routing_policy"]["processing_config_contract"] = (
        "pocketfinancer.processing-config/6"
    )
    manifest["compatibility"]["frozen_release_ids"].append("native-integration-v5")
    manifest["artifacts"].extend(
        {"path": path, "contract": contract, "sha256": file_sha256(REPO_ROOT / path)}
        for path, contract in NEW_ARTIFACTS
    )
    jsonschema.validate(
        manifest,
        _read_json(
            "configs/sms_processing/contracts/releases/v6/release-manifest.schema.json"
        ),
    )
    return manifest


def main() -> None:
    print(json.dumps(build_manifest(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
