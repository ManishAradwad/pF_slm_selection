from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.probe_lfm25_lora_memory import (
    QUANTIZATION_MODES,
    _input_provenance,
    _output_path_is_ignored,
    parse_args,
    _resolve_safe_output_path,
    target_module_inventory,
)


def _required() -> list[str]:
    return ["--model", "model", "--train", "train.jsonl"]


def test_probe_defaults_to_real_bf16_lora_capacity_gate() -> None:
    args = parse_args(_required())

    assert QUANTIZATION_MODES == ("bf16", "qlora-nf4")
    assert args.quantization_mode == "bf16"
    assert args.prompt_profile == "android"
    assert args.batch_size == 1
    assert args.max_length == 2304
    assert args.rank == 16
    assert args.alpha == 32
    assert args.first_supervised_token_weight == 3.0


def test_probe_accepts_explicit_nf4_qlora_and_rejects_invalid_limits() -> None:
    args = parse_args([*_required(), "--quantization-mode", "qlora-nf4"])
    assert args.quantization_mode == "qlora-nf4"

    with pytest.raises(SystemExit):
        parse_args([*_required(), "--batch-size", "0"])
    with pytest.raises(SystemExit):
        parse_args([*_required(), "--dropout", "1"])
    with pytest.raises(SystemExit):
        parse_args([*_required(), "--max-length", "3073"])


def test_target_inventory_reports_only_aggregate_leaf_coverage() -> None:
    class FakeModel:
        @staticmethod
        def named_modules():
            return iter(
                (
                    ("", object()),
                    ("model.layers.0.in_proj", object()),
                    ("model.layers.1.q_proj", object()),
                    ("model.layers.1.k_proj", object()),
                    ("model.layers.1.unrelated", object()),
                )
            )

    inventory = target_module_inventory(FakeModel())

    assert inventory["matched_module_count"] == 3
    assert inventory["matched_leaf_counts"]["in_proj"] == 1
    assert inventory["matched_leaf_counts"]["q_proj"] == 1
    assert inventory["matched_leaf_counts"]["k_proj"] == 1
    assert "model.layers.0.in_proj" not in str(inventory)


def test_probe_resolves_relative_results_output_before_checking_ignore() -> None:
    assert _output_path_is_ignored(Path("RESULTS/pocketfinancer/probes/probe.json"))
    assert not _output_path_is_ignored(Path("../outside-repository/probe.json"))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_probe_output_fails_closed_for_unignored_outside_and_tracked_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    tracked = repo / "ignored" / "tracked.json"
    tracked.parent.mkdir()
    tracked.write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "add", "--force", "ignored/tracked.json")

    ignored_output = repo / "ignored" / "new.json"
    assert _resolve_safe_output_path(ignored_output, repo) == ignored_output.resolve()

    with pytest.raises(ValueError, match="ignored path inside the repository"):
        _resolve_safe_output_path(repo / "visible.json", repo)
    with pytest.raises(ValueError, match="ignored path inside the repository"):
        _resolve_safe_output_path(tmp_path / "outside.json", repo)
    with pytest.raises(ValueError, match="must not overwrite a tracked"):
        _resolve_safe_output_path(tracked, repo)


def test_probe_input_provenance_binds_file_and_resolved_contract(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    contents = b'{"synthetic":true}\n'
    train.write_bytes(contents)

    provenance = _input_provenance(train, "pocketfinancer")

    assert provenance["train_file"] == {
        "path": str(train.resolve()),
        "bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }
    serialization = provenance["serialization"]
    assert serialization["prompt_profile"] == "android"
    assert serialization["contract"]["profile"] == "pocketfinancer"
    assert len(serialization["contract"]["prompt_template_sha256"]) == 64
    assert "synthetic" not in str(provenance)
