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
def test_innocuous_names_cannot_bypass_protected_top_level_trees(path: str, tree: str) -> None:
    assert find_violations([path]) == [Violation(path, f"protected output tree: {tree}")]


def test_protected_tree_names_are_exact_and_top_level_only() -> None:
    paths = ["models/runtime.py", "private_data/schema.py", "src/PRIVATE_DATA/schema.py"]

    assert find_violations(paths) == []


def test_safe_tracked_paths_and_fixture_exception_are_allowed() -> None:
    paths = [
        "DATA/extraction_ds.jsonl",
        "DATA/sms_extraction.yaml",
        "export_sms.py",
        "models/runtime.py",
    ]

    assert find_violations(paths) == []


def test_versioned_candidate_protocol_artifacts_have_exact_exceptions() -> None:
    paths = [
        "DATA/candidate_protocol_v1_golden.json",
        "DATA/annotation_component_v1_synthetic.jsonl",
        "configs/contracts/pocketfinancer-candidate-v1.json",
        "configs/pipelines/pocketfinancer-lfm2.5-350m-candidate-v1.json",
        "configs/prompts/pocketfinancer-android-gemma-candidate-v2-v1.json",
        "configs/prompts/pocketfinancer-android-qwen-candidate-v2-v1.json",
    ]

    assert find_violations(paths) == []

    near_misses = [
        "DATA/candidate_protocol_v1_golden-copy.json",
        "DATA/annotation_component_v1_synthetic-copy.jsonl",
        "configs/contracts/pocketfinancer-candidate-v1-copy.json",
        "configs/pipelines/pocketfinancer-lfm2.5-350m-candidate-v1-copy.json",
        "configs/prompts/pocketfinancer-android-gemma-candidate-v2-v1-copy.json",
        "configs/prompts/pocketfinancer-android-qwen-candidate-v2-v1-copy.json",
    ]

    assert find_violations(near_misses) == [
        Violation(near_misses[0], "raw export"),
        Violation(near_misses[1], "raw export"),
        Violation(near_misses[2], "raw export"),
        Violation(near_misses[3], "raw export"),
        Violation(near_misses[4], "raw export"),
        Violation(near_misses[5], "raw export"),
    ]


def test_grandfather_exception_is_exact() -> None:
    assert find_violations(["data/extraction_ds.jsonl"]) == [
        Violation("data/extraction_ds.jsonl", "raw export")
    ]


def test_removed_analysis_artifacts_are_rejected_at_root() -> None:
    assert find_violations(["error_analysis.txt", "results_analysis.ipynb"]) == [
        Violation("error_analysis.txt", "result artifact"),
        Violation("results_analysis.ipynb", "result artifact"),
    ]


def test_windows_separators_cannot_bypass_policy() -> None:
    assert find_violations([r"RESULTS\demo\metrics.json"]) == [
        Violation("RESULTS/demo/metrics.json", "result artifact")
    ]
