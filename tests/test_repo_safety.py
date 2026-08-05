import pytest

from scripts.check_repo_safety import Violation, find_violations


@pytest.mark.parametrize(
    ("path", "category"),
    [
        ("private/messages.sqlite3", "database"),
        ("private/messages.db-wal", "database"),
        ("all_sms.csv", "tabular export"),
        ("exports/messages.json", "raw export"),
        ("DATA/another_dataset.jsonl", "raw export"),
        ("MODELS/demo.gguf", "model weight"),
        ("checkpoints/run-1/state.json", "checkpoint"),
        ("RESULTS/demo/samples_sms.jsonl", "result artifact"),
        ("logs/evaluation.log", "result artifact"),
        ("archive/error_analysis.txt", "result artifact"),
        ("archive/results_analysis.ipynb", "result artifact"),
        ("candidate.jsonl", "raw export"),
        ("scripts/example.py.orig", "patch backup"),
    ],
)
def test_private_and_generated_paths_are_rejected(path: str, category: str) -> None:
    assert find_violations([path]) == [Violation(path, category)]


@pytest.mark.parametrize(
    ("path", "tree"),
    [
        ("PRIVATE_DATA/lfm25/split_manifest.jsonl", "PRIVATE_DATA"),
        ("PUBLIC_CANDIDATE/lfm25/safe_preview.jsonl", "PUBLIC_CANDIDATE"),
        ("TRAINING_ARTIFACTS/lfm25/seed-29/README.md", "TRAINING_ARTIFACTS"),
        ("MODELS/LFM2.5-350M/tokenizer_config.json", "MODELS"),
        ("RESULTS/lfm25/benchmarks/notes.md", "RESULTS"),
        ("UPSTREAM/llama.cpp/examples/example.py", "UPSTREAM"),
    ],
)
def test_innocuous_names_cannot_bypass_protected_top_level_trees(
    path: str, tree: str
) -> None:
    assert find_violations([path]) == [Violation(path, f"protected output tree: {tree}")]


def test_protected_tree_names_are_exact_and_top_level_only() -> None:
    paths = ["models/runtime.py", "private_data/schema.py", "src/PRIVATE_DATA/schema.py"]

    assert find_violations(paths) == []


def test_explicit_legacy_paths_are_allowed() -> None:
    paths = [
        "DATA/extraction_ds.jsonl",
        "DATA/sms_extraction.yaml",
        "error_analysis.txt",
        "export_sms.py",
        "models/runtime.py",
        "results_analysis.ipynb",
    ]

    assert find_violations(paths) == []


def test_grandfather_exception_is_exact() -> None:
    assert find_violations(["data/extraction_ds.jsonl"]) == [
        Violation("data/extraction_ds.jsonl", "raw export")
    ]


def test_legacy_analysis_exceptions_are_exact() -> None:
    assert find_violations(["other/error_analysis.txt", "other/results_analysis.ipynb"]) == [
        Violation("other/error_analysis.txt", "result artifact"),
        Violation("other/results_analysis.ipynb", "result artifact"),
    ]


def test_windows_separators_cannot_bypass_policy() -> None:
    assert find_violations([r"RESULTS\demo\metrics.json"]) == [
        Violation("RESULTS/demo/metrics.json", "result artifact")
    ]
