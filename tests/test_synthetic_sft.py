import json
from pathlib import Path
import subprocess

import pytest

from lfm25.public_candidate import audit_candidate_rows, generate_candidate_rows
from lfm25.synthetic_sft import (
    DEFAULT_HOLDOUT_FAMILIES,
    SFT_PROVENANCE,
    ensure_ignored_private_output,
    split_by_template_family,
    write_synthetic_sft_artifacts,
)


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    output = repo / "PRIVATE_DATA" / "lfm25"
    return repo, output


def test_split_holds_out_complete_transaction_and_negative_families() -> None:
    rows = generate_candidate_rows(count=180, seed=808)
    audited = audit_candidate_rows(rows, private_texts=[])
    train, dev = split_by_template_family(audited.accepted_rows)

    train_families = {row["template_family"] for row in train}
    dev_families = {row["template_family"] for row in dev}
    assert train_families.isdisjoint(dev_families)
    assert dev_families == set(DEFAULT_HOLDOUT_FAMILIES)
    assert any(row["class"].startswith("hard_negative_") for row in dev)
    assert any(not row["class"].startswith("hard_negative_") for row in dev)
    assert all(row["sft_label_provenance"] == SFT_PROVENANCE for row in train + dev)


def test_private_output_guard_fails_closed_until_ignored(tmp_path: Path) -> None:
    repo, output = _repo(tmp_path)
    with pytest.raises(RuntimeError, match="not protected"):
        ensure_ignored_private_output(repo, output)
    (repo / ".gitignore").write_text("PRIVATE_DATA/\n", encoding="utf-8")
    ensure_ignored_private_output(repo, output)
    with pytest.raises(RuntimeError, match="must stay under"):
        ensure_ignored_private_output(repo, repo / "elsewhere")


def test_writer_persists_only_synthetic_rows_and_aggregate_manifest(
    tmp_path: Path,
) -> None:
    repo, output = _repo(tmp_path)
    (repo / ".gitignore").write_text("PRIVATE_DATA/\n", encoding="utf-8")
    rows = generate_candidate_rows(count=36, seed=909)
    audit = audit_candidate_rows(rows, private_texts=["private sentinel never persist"])
    train, dev = split_by_template_family(audit.accepted_rows)

    paths = write_synthetic_sft_artifacts(
        train,
        dev,
        repo_root=repo,
        output_dir=output,
        seed=909,
        generated_rows=len(rows),
        audit_report=audit.report,
        holdout_families=DEFAULT_HOLDOUT_FAMILIES,
    )

    assert {path.name for path in paths} == {
        "synthetic_sft_train.jsonl",
        "synthetic_sft_dev.jsonl",
        "synthetic_sft_manifest.json",
    }
    assert all("private sentinel" not in path.read_text(encoding="utf-8") for path in paths)
    manifest = json.loads((output / "synthetic_sft_manifest.json").read_text())
    assert manifest["template_family_overlap"] == []
    assert manifest["release_authorized"] is False
    assert manifest["label_provenance"] == SFT_PROVENANCE
    assert manifest["train_rows"] + manifest["dev_rows"] == 36
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_synthetic_sft_artifacts(
            train,
            dev,
            repo_root=repo,
            output_dir=output,
            seed=909,
            generated_rows=len(rows),
            audit_report=audit.report,
            holdout_families=DEFAULT_HOLDOUT_FAMILIES,
        )
