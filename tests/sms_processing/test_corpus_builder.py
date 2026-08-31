"""Synthetic-only corpus completeness, grouping, privacy, and leakage tests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pocketfinancer_sms.corpus.manifest import build_private_corpus
from pocketfinancer_sms.corpus.pools import PoolInput, assign_pools, leakage_audit


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_template_components_never_cross_pool_boundaries() -> None:
    rows = [
        PoolInput("s1", "2024-01-01T00:00:00Z", "e1", "t1", "st1"),
        PoolInput("s2", "2024-02-01T00:00:00Z", "e2", "t1", "st2"),
        PoolInput("s3", "2026-01-01T00:00:00Z", "e3", "t2", "st3"),
        PoolInput("s4", "2024-01-01T00:00:00Z", "e4", "t3", "st4"),
    ]
    assignments = assign_pools(
        rows,
        regression_template_hashes={"t3"},
        legacy_template_hashes=set(),
        later_time_cutoff="2025-09-01T00:00:00Z",
    )

    assert assignments["s1"] == assignments["s2"]
    assert assignments["s3"] == "later_time_holdout"
    assert assignments["s4"] == "regression_only"
    assert leakage_audit(rows, assignments)["passed"] is True


def test_synthetic_corpus_build_retains_every_row_and_writes_private_permissions(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("PRIVATE_DATA/\n*.json\n", encoding="utf-8")
    source = [
        {
            "id": 1,
            "date": "2024-01-01T10:00:00Z",
            "sender": "SYNTH-BANK",
            "text": "INR 10 was debited from account **1111 at STORE.",
            "is_from_me": False,
            "service": "SMS",
        },
        {
            "id": 2,
            "date": "2024-01-02T10:00:00Z",
            "sender": "SYNTH-BANK",
            "text": "123456 is your OTP for login.",
            "is_from_me": False,
            "service": "SMS",
        },
        {
            "id": 3,
            "date": "2026-01-02T10:00:00Z",
            "sender": "SYNTH-BANK",
            "text": "EUR 4 was credited to account **2222 from FRIEND.",
            "is_from_me": False,
            "service": "SMS",
        },
        {
            "id": 4,
            "date": "2024-01-03T10:00:00Z",
            "sender": "SYNTH-SELF",
            "text": "INR 2 was paid at STORE.",
            "is_from_me": True,
            "service": "SMS",
        },
    ]
    _write_json(repo / "synthetic_source.json", source)
    _write_jsonl(repo / "fixture.jsonl", [{"sms": source[0]["text"]}])
    _write_jsonl(
        repo / "PRIVATE_DATA" / "legacy.jsonl",
        [
            {
                "sms": source[1]["text"],
                "sender": source[1]["sender"],
                "decision": "not_transaction",
            }
        ],
    )
    config = {
        "contract": "pocketfinancer.corpus-run/1",
        "source_path": "synthetic_source.json",
        "expected_source_rows": 4,
        "output_root": "PRIVATE_DATA/sms_processing",
        "primary_currency": "INR",
        "profile_ids": ["core-en", "india"],
        "later_time_cutoff": "2025-09-01T00:00:00Z",
        "source_id_key_path": "PRIVATE_DATA/sms_processing/.source-id-key",
        "regression_fixture_path": "fixture.jsonl",
        "legacy_review_path": "PRIVATE_DATA/legacy.jsonl",
        "offline_retention": "retain_every_source_row",
        "build_sft_targets": False,
    }
    _write_json(repo / "config.json", config)

    summary = build_private_corpus(repo, Path("config.json"))
    current = json.loads(
        (repo / "PRIVATE_DATA" / "sms_processing" / "CURRENT.json").read_text(encoding="utf-8")
    )
    run = repo / "PRIVATE_DATA" / "sms_processing" / "runs" / current["run_id"]
    manifest = [
        json.loads(line)
        for line in (run / "canonical_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["row_count"] == 4
    assert len(manifest) == 4
    assert len({row["source_id"] for row in manifest}) == 4
    assert sum(summary["pool_counts"].values()) == 4
    assert (run / "reports" / "leakage_audit.json").is_file()
    assert os.stat(run).st_mode & 0o777 == 0o700
    assert os.stat(run / "canonical_manifest.jsonl").st_mode & 0o777 == 0o600

    repeated = build_private_corpus(repo, Path("config.json"))
    assert repeated == summary
