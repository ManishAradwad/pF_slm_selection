import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from lfm25.public_candidate import (
    ALL_TEMPLATES,
    FORBIDDEN_LINKAGE_KEYS,
    LABEL_FIELDS,
    MANUAL_REVIEW_STATUS,
    audit_candidate_rows,
    ensure_ignored_output_tree,
    generate_candidate_rows,
    load_private_text_sources,
    load_private_texts_export,
    load_private_texts_sqlite,
    scan_sensitive_tokens,
    validate_candidate_row,
    write_audit_artifacts,
)
from scripts.audit_lfm25_public_candidate import main as audit_main
from scripts.audit_lfm25_public_candidate import parse_args as audit_parse_args
from scripts.build_lfm25_synthetic_sft import parse_args as synthetic_parse_args
from scripts.probe_lfm25_memorization import parse_args as probe_parse_args


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def _init_ignored_repo(root: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("PUBLIC_CANDIDATE/\n", encoding="utf-8")
    return root / "PUBLIC_CANDIDATE" / "lfm25"


def test_generator_is_deterministic_diverse_and_uses_four_field_labels() -> None:
    count = len(ALL_TEMPLATES) * 2
    first = generate_candidate_rows(count=count, seed=101)
    second = generate_candidate_rows(count=count, seed=101)

    assert first == second
    assert len({row["public_id"] for row in first}) == count
    assert len({row["sender"] for row in first}) >= 5
    assert {row["template_family"] for row in first} == {
        template.family for template in ALL_TEMPLATES
    }
    assert all(not validate_candidate_row(row) for row in first)
    assert all(row["manual_review"] == MANUAL_REVIEW_STATUS for row in first)
    assert all(not (_all_keys(row) & FORBIDDEN_LINKAGE_KEYS) for row in first)

    for row in first:
        label = json.loads(row["expected"])
        if row["class"].startswith("hard_negative_"):
            assert label is None
        else:
            assert set(label) == set(LABEL_FIELDS)
            assert label["type"] in {"debit", "credit"}


def test_generator_owned_safe_tokens_pass_but_realistic_phone_is_blocked() -> None:
    rows = generate_candidate_rows(count=len(ALL_TEMPLATES), seed=202)
    for row in rows:
        findings = scan_sensitive_tokens(row)
        assert all(result["blocked"] == 0 for result in findings.values())

    unsafe = dict(rows[0])
    unsafe["sms"] += " Call +91 98765 43210."
    bundle = audit_candidate_rows([unsafe], private_texts=[])

    assert not bundle.accepted_rows
    assert bundle.report["rejected_counts"] == {"sensitive_token": 1}
    assert bundle.report["sensitive_data_scan"]["phone"]["blocked"] == 1


def test_exact_private_collision_is_rewritten_without_retaining_private_text() -> None:
    row = generate_candidate_rows(count=1, seed=303)[0]
    private_sentinel = "private sentinel phrase alpha beta gamma delta epsilon"
    colliding = dict(row)
    colliding["sms"] = private_sentinel

    bundle = audit_candidate_rows([colliding], [private_sentinel])
    serialized_report = json.dumps(bundle.report, sort_keys=True)
    serialized_manifest = json.dumps(bundle.memorization_probe_manifest, sort_keys=True)

    assert len(bundle.accepted_rows) == 1
    assert bundle.accepted_rows[0]["sms"] != private_sentinel
    assert bundle.accepted_rows[0]["manual_review"] == MANUAL_REVIEW_STATUS
    assert bundle.report["private_similarity_audit"]["exact_matches_before_rewrite"] == 1
    assert bundle.report["private_similarity_audit"]["rows_rewritten"] == 1
    assert private_sentinel not in serialized_report
    assert private_sentinel not in serialized_manifest


def test_duplicate_and_invalid_label_rejections_are_aggregate_only() -> None:
    rows = generate_candidate_rows(count=2, seed=404)
    duplicate = dict(rows[0])
    invalid = dict(rows[1])
    invalid["expected"] = json.dumps({"amount": None})

    bundle = audit_candidate_rows([rows[0], duplicate, invalid], private_texts=[])

    assert len(bundle.accepted_rows) == 1
    assert bundle.report["rejected_counts"] == {
        "duplicate_public_id": 1,
        "schema_or_label": 1,
    }
    assert bundle.report["schema_and_label_audit"]["error_counts"] == {"label_fields": 1}


def test_output_guard_fails_closed_until_public_tree_is_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    output = repo / "PUBLIC_CANDIDATE" / "lfm25"

    with pytest.raises(RuntimeError, match="not protected"):
        ensure_ignored_output_tree(repo, output)

    (repo / ".gitignore").write_text("PUBLIC_CANDIDATE/\n", encoding="utf-8")
    ensure_ignored_output_tree(repo, output)

    with pytest.raises(RuntimeError, match="must stay under"):
        ensure_ignored_output_tree(repo, repo / "elsewhere")


def test_written_audit_artifacts_are_pending_and_do_not_persist_private_text(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = _init_ignored_repo(repo)
    private_sentinel = "private archive sentence quartz violet seven maple"
    row = generate_candidate_rows(count=1, seed=505)[0]
    row["sms"] = private_sentinel
    bundle = audit_candidate_rows([row], [private_sentinel])

    paths = write_audit_artifacts(
        bundle,
        repo_root=repo,
        output_dir=output,
    )

    assert {path.name for path in paths} == {
        "candidate.jsonl",
        "safe_preview.jsonl",
        "audit_report.json",
        "memorization_probe_manifest.json",
        "dataset_card.md",
        "license_data_rights_review.md",
    }
    assert all(private_sentinel not in path.read_text(encoding="utf-8") for path in paths)
    card = (output / "dataset_card.md").read_text(encoding="utf-8")
    assert "## Detailed audited coverage" in card
    assert "### Classes" in card
    assert "### Template families" in card
    assert "## Prohibited uses" in card
    assert "## Known biases and limitations" in card
    assert "high-impact decision" in card
    assert "Do not publish, redistribute" in card
    assert "English-language and India-focused" in card
    assert "passing aggregate audit does not prove" in card
    for coverage_name in ("classes", "template_families"):
        for name, count in bundle.report["coverage"][coverage_name].items():
            assert f"| `{name}` | {count} |" in card
    candidate = json.loads((output / "candidate.jsonl").read_text(encoding="utf-8"))
    assert candidate["manual_review"] == MANUAL_REVIEW_STATUS
    assert bundle.report["release_decision"] == "not_made"
    assert bundle.report["license_decision"] == "not_made"


def test_audit_cli_does_not_print_or_mutate_private_text(tmp_path: Path, capsys) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = _init_ignored_repo(repo)
    private_sentinel = "private cli sentinel indigo cedar twelve orbit"
    row = generate_candidate_rows(count=1, seed=606)[0]
    row["sms"] = private_sentinel
    candidate_path = output / "candidate_unreviewed.jsonl"
    output.mkdir(parents=True)
    candidate_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    private_path = repo / "private.jsonl"
    private_path.write_text(
        json.dumps({"sms": private_sentinel}) + "\n",
        encoding="utf-8",
    )
    before = private_path.read_bytes()

    exit_code = audit_main(
        [
            "--repo-root",
            str(repo),
            "--candidate",
            str(candidate_path),
            "--private-jsonl",
            str(private_path),
            "--output-dir",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert private_sentinel not in captured.out
    assert private_sentinel not in captured.err
    assert private_path.read_bytes() == before


def test_sqlite_loader_is_read_only_and_returns_only_text(tmp_path: Path) -> None:
    database = tmp_path / "private.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE message (id INTEGER PRIMARY KEY, text TEXT, sender TEXT)")
    connection.executemany(
        "INSERT INTO message(text, sender) VALUES (?, ?)",
        [("private one", "sender-a"), ("", "sender-b"), (None, "sender-c")],
    )
    connection.commit()
    connection.close()
    before = database.read_bytes()

    assert load_private_texts_sqlite([database]) == ["private one"]
    assert database.read_bytes() == before
    with pytest.raises(ValueError, match="simple identifiers"):
        load_private_texts_sqlite([database], table="message; DROP TABLE message")


def test_complete_json_export_loader_covers_every_row_and_only_returns_text(
    tmp_path: Path,
) -> None:
    export = tmp_path / "all_sms.json"
    rows = [
        {"id": 1, "text": "private export one", "metadata": "do not return alpha"},
        {"id": 2, "text": "private export two", "metadata": "do not return beta"},
        {"id": 3, "text": "private export three", "metadata": "do not return gamma"},
    ]
    export.write_text(json.dumps(rows), encoding="utf-8")
    before = export.read_bytes()

    assert load_private_texts_export([export]) == [row["text"] for row in rows]
    assert export.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        ["private invalid row sentinel"],
        [{"other": "private missing field sentinel"}],
        [{"text": 42}],
        [{"text": "   "}],
    ],
)
def test_complete_json_export_loader_fails_safely_on_invalid_rows(
    tmp_path: Path,
    payload: object,
) -> None:
    export = tmp_path / "all_sms.json"
    export.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        load_private_texts_export([export])

    assert "sentinel" not in str(caught.value)


def test_complete_csv_export_and_mixed_source_aggregation(tmp_path: Path) -> None:
    jsonl = tmp_path / "private.jsonl"
    jsonl.write_text(json.dumps({"sms": "jsonl private"}) + "\n", encoding="utf-8")
    export = tmp_path / "all_sms.csv"
    export.write_text(
        "id,text,metadata\n1,csv private one,hidden-a\n2,csv private two,hidden-b\n",
        encoding="utf-8",
    )
    database = tmp_path / "private.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE message (text TEXT, metadata TEXT)")
    connection.execute(
        "INSERT INTO message(text, metadata) VALUES (?, ?)",
        ("sqlite private", "hidden-c"),
    )
    connection.commit()
    connection.close()

    assert load_private_text_sources(
        jsonl_paths=[jsonl],
        export_paths=[export],
        sqlite_paths=[database],
    ) == [
        "jsonl private",
        "csv private one",
        "csv private two",
        "sqlite private",
    ]


def test_all_private_cli_parsers_accept_repeatable_export_only_sources(
    tmp_path: Path,
) -> None:
    first = tmp_path / "all_sms.json"
    second = tmp_path / "all_sms.csv"
    source_args = ["--private-export", str(first), "--private-export", str(second)]

    audit_args = audit_parse_args(source_args)
    synthetic_args = synthetic_parse_args(source_args)
    probe_args = probe_parse_args(
        [
            "--model",
            str(tmp_path / "model"),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--dev",
            str(tmp_path / "dev.jsonl"),
            *source_args,
        ]
    )

    assert audit_args.private_export == [first, second]
    assert synthetic_args.private_export == [first, second]
    assert probe_args.private_export == [first, second]
    assert {
        audit_args.private_export_text_field,
        synthetic_args.private_export_text_field,
        probe_args.private_export_text_field,
    } == {"text"}


def test_audit_cli_accepts_immutable_sqlite_without_printing_text(
    tmp_path: Path,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = _init_ignored_repo(repo)
    private_sentinel = "private sqlite sentinel amber spruce eight comet"
    row = generate_candidate_rows(count=1, seed=707)[0]
    row["sms"] = private_sentinel
    candidate_path = output / "candidate_unreviewed.jsonl"
    output.mkdir(parents=True)
    candidate_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    database = repo / "private.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE message (text TEXT)")
    connection.execute("INSERT INTO message(text) VALUES (?)", (private_sentinel,))
    connection.commit()
    connection.close()
    before = database.read_bytes()

    exit_code = audit_main(
        [
            "--repo-root",
            str(repo),
            "--candidate",
            str(candidate_path),
            "--private-sqlite",
            str(database),
            "--output-dir",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert private_sentinel not in captured.out
    assert private_sentinel not in captured.err
    assert database.read_bytes() == before


def test_audit_cli_accepts_export_only_without_printing_text(
    tmp_path: Path,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = _init_ignored_repo(repo)
    private_sentinel = "private export sentinel ultramarine willow nine asteroid"
    row = generate_candidate_rows(count=1, seed=808)[0]
    row["sms"] = private_sentinel
    candidate_path = output / "candidate_unreviewed.jsonl"
    output.mkdir(parents=True)
    candidate_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    export = repo / "all_sms.json"
    export.write_text(
        json.dumps([{"id": 1, "text": private_sentinel, "sender": "hidden"}]),
        encoding="utf-8",
    )
    before = export.read_bytes()

    exit_code = audit_main(
        [
            "--repo-root",
            str(repo),
            "--candidate",
            str(candidate_path),
            "--private-export",
            str(export),
            "--output-dir",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert private_sentinel not in captured.out
    assert private_sentinel not in captured.err
    assert export.read_bytes() == before
