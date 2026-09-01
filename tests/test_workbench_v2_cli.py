from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_pocketfinancer_workbench_v2.py"
FIXTURE = REPOSITORY_ROOT / "tests/fixtures/pocketfinancer_workbench_v2_synthetic.json"


def test_cli_validates_exact_invented_fixture_without_row_output() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "validate", str(FIXTURE), "--invented-fixture"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)

    assert output["status"] == "valid"
    assert output["counts"]["rows"] == 4
    assert "Orbit Bank" not in completed.stdout
    assert "XX1234" not in completed.stdout


def test_cli_refuses_same_fixture_without_explicit_invented_gate() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "validate", str(FIXTURE)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "PRIVATE_DATA" in completed.stderr
    assert "Orbit Bank" not in completed.stderr
