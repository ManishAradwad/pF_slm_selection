"""Fake-provider coverage for the macOS-only Apple SDK evaluation lane."""

from __future__ import annotations

import asyncio
import fcntl
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from pocketfinancer_sms import apple_evaluation, cli, evaluation
from pocketfinancer_sms.apple_evaluation import (
    AppleFMProvider,
    _map_guided_result,
    evaluate_apple_foundation_models,
)
from pocketfinancer_sms.evaluation import EvaluationInterrupted, ExtractorEvaluationError


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/sms_processing/fixtures/extractor/synthetic-suite.jsonl"
SCHEMA = ROOT / "configs/sms_processing/evaluations/apple-fm-flat-v1.schema.json"


def _cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]


def _flat_gold(case: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        key: "" if key in apple_evaluation._STRING_FIELDS else 0 for key in apple_evaluation._FIELDS
    }
    gold = case["gold"]
    value["decision"] = gold["decision"]
    if gold["decision"] != "posted":
        return value
    for field in ("amount", "direction", "account", "counterparty"):
        content = gold.get(field)
        if content is None:
            continue
        if field == "amount":
            value["amount_value"] = content["value"]
            value["amount_currency"] = content["currency"]
        elif field == "direction":
            value["direction_value"] = content["value"]
        elif field == "account":
            value["account_reference"] = content["reference"]
        else:
            value["counterparty_value"] = content["value"]
        span = content["evidence"]
        value[f"{field}_start"] = span["start_scalar"]
        value[f"{field}_end"] = span["end_scalar"]
        value[f"{field}_text"] = span["text"]
    return value


class FakeProvider:
    provider_id = "fake-apple-provider"
    runtime_version = "fake-sdk-0"

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def infer(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _allow_tmp_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        apple_evaluation, "require_private_output", lambda _root, path: path.resolve()
    )
    monkeypatch.setattr(evaluation, "require_private_output", lambda _root, path: path.resolve())


def _evaluate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: FakeProvider,
    *,
    output_name: str = "run",
    locale: str = "en_US",
) -> dict[str, Any]:
    _allow_tmp_private(monkeypatch)
    return evaluate_apple_foundation_models(
        ROOT,
        suite="synthetic",
        output_dir=tmp_path / output_name,
        provider=provider,
        locale=locale,
    )


def test_fake_provider_scores_same_synthetic_suite_and_keeps_rows_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = _cases()
    provider = FakeProvider([_flat_gold(case) for case in cases])
    summary = _evaluate(monkeypatch, tmp_path, provider)

    assert capsys.readouterr().out == ""
    assert summary["lane"] == "apple_fm_python"
    assert summary["total_rows"] == len(cases)
    assert summary["metrics"]["classification"]["accuracy"] == 1.0
    assert summary["metrics"]["transaction"]["f1"] == 1.0
    assert summary["metrics"]["strict_posted_exact_success"]["rate"] == 1.0
    assert summary["metrics"]["source_grounding"]["rejected_count"] == 0
    assert summary["metrics"]["abstention"]["correct_on_gold_abstain"] == 1
    assert len(provider.calls) == len(cases)
    assert provider.calls[0]["schema"] == json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert json.loads(provider.calls[0]["prompt"])["contract"] == (
        "pocketfinancer.sms-extractor-input/1"
    )
    report_path = tmp_path / "run/report.json"
    checkpoint_path = tmp_path / "run/checkpoint.json"
    report_text = report_path.read_text(encoding="utf-8")
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    assert cases[0]["source"] not in report_text
    assert '"guided_output"' not in report_text
    assert cases[0]["gold"]["amount"]["evidence"]["text"] in checkpoint_text
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o600
    report = json.loads(report_text)
    runtime = report["provenance"]["runtime"]
    assert runtime["simulated"] is True
    assert runtime["model_file_sha256"] is None
    assert report["provenance"]["assets"]["suite_manifest_sha256"]
    assert report["provenance"]["assets"]["prompt_sha256"]
    assert report["provenance"]["assets"]["schema_sha256"]
    assert report["provenance"]["scoring_version"]
    assert "non-causal" in report["contract_differences"]["comparison"]


def test_grounding_abstention_and_failures_are_deterministic_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = _cases()
    outputs: list[Any] = [_flat_gold(case) for case in cases]
    outputs[0] = {**outputs[0], "amount_text": "PRIVATE_INVALID_SPAN"}
    outputs[1] = RuntimeError("PRIVATE_RUNTIME_PROMPT_ECHO")
    outputs[2] = {**_flat_gold(cases[7]), "decision": "abstain"}
    outputs[3] = {**outputs[3], "unexpected": "PRIVATE_UNKNOWN_FIELD"}
    summary = _evaluate(monkeypatch, tmp_path, FakeProvider(outputs))

    metrics = summary["metrics"]
    assert metrics["source_grounding"]["rejected_count"] == 1
    assert metrics["failures"] == {
        "malformed_count": 2,
        "runtime_failure_count": 1,
        "total_count": 3,
    }
    assert metrics["classification"]["missed_transactions"] == 4
    assert metrics["abstention"]["predicted_count"] == 2
    report_text = (tmp_path / "run/report.json").read_text(encoding="utf-8")
    assert "PRIVATE_INVALID_SPAN" not in report_text
    assert "PRIVATE_RUNTIME_PROMPT_ECHO" not in report_text
    assert "PRIVATE_UNKNOWN_FIELD" not in report_text
    assert metrics["reason_code_counts"] == {
        "apple_guided_result_invalid": 1,
        "apple_runtime_failure": 1,
        "extractor_evidence_mismatch": 1,
    }


def test_interruption_resume_mismatch_and_duplicate_case_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = _cases()
    outputs = [_flat_gold(case) for case in cases]
    first = FakeProvider([outputs[0], KeyboardInterrupt()])
    with pytest.raises(EvaluationInterrupted) as caught:
        _evaluate(monkeypatch, tmp_path, first)
    assert caught.value.completed_rows == 1
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "interrupted"
    assert len(checkpoint["records"]) == 1
    stale_temp = tmp_path / "run/.checkpoint.json.abcd1234"
    stale_temp.write_text("partial", encoding="utf-8")

    with pytest.raises(ExtractorEvaluationError, match="provenance does not match"):
        _evaluate(monkeypatch, tmp_path, FakeProvider([]), locale="en_GB")
    resumed = FakeProvider(outputs[1:])
    assert _evaluate(monkeypatch, tmp_path, resumed)["status"] == "complete"
    assert not stale_temp.exists()
    assert len(resumed.calls) == len(cases) - 1
    assert _evaluate(monkeypatch, tmp_path, FakeProvider([]))["status"] == "complete"

    original = apple_evaluation._load_suite

    def duplicated(*args: Any, **kwargs: Any) -> Any:
        rows, capabilities, fingerprint = original(*args, **kwargs)
        return [*rows, rows[0]], capabilities, fingerprint

    monkeypatch.setattr(apple_evaluation, "_load_suite", duplicated)
    with pytest.raises(ExtractorEvaluationError, match="duplicate case IDs"):
        _evaluate(monkeypatch, tmp_path, FakeProvider([]), output_name="duplicate")


def test_concurrent_run_and_mismatched_report_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_private(monkeypatch)
    output = tmp_path / "locked"
    output.mkdir()
    with (output / "run.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ExtractorEvaluationError, match="already in use"):
            _evaluate(monkeypatch, tmp_path, FakeProvider([]), output_name="locked")
    cases = _cases()
    _evaluate(monkeypatch, tmp_path, FakeProvider([_flat_gold(case) for case in cases]))
    report_path = tmp_path / "run/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["configuration_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExtractorEvaluationError, match="report provenance does not match"):
        _evaluate(monkeypatch, tmp_path, FakeProvider([]))


def test_live_wsl_attempt_is_clear_and_does_not_create_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if sys.platform == "darwin":
        pytest.skip("WSL guard applies only outside macOS")
    output = tmp_path / "should-not-exist"
    with pytest.raises(ExtractorEvaluationError, match="requires compatible macOS"):
        evaluate_apple_foundation_models(ROOT, suite="synthetic", output_dir=output)
    assert not output.exists()
    assert capsys.readouterr().out == ""
    code = cli.main(["--repo-root", str(ROOT), "evaluate-apple-fm", "--output-dir", str(output)])
    assert code == 2
    console = json.loads(capsys.readouterr().out)
    assert "requires compatible macOS" in console["reason"]
    assert not output.exists()


def test_sdk_adapter_uses_official_guided_response_api_with_fake_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class FakeModel:
        def is_available(self) -> tuple[bool, None]:
            return True, None

    class FakeContent:
        is_complete = True

        def to_json(self) -> str:
            return json.dumps(_flat_gold(_cases()[0]))

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            seen["session"] = kwargs

        async def respond(self, prompt: str, **kwargs: Any) -> FakeContent:
            seen["prompt"] = prompt
            seen["respond"] = kwargs
            return FakeContent()

    class FakeSampling:
        @staticmethod
        def greedy() -> str:
            return "greedy"

    class FakeOptions:
        def __init__(self, **kwargs: Any) -> None:
            seen["options"] = kwargs

    class FakeSDK:
        SystemLanguageModel = FakeModel
        LanguageModelSession = FakeSession
        SamplingMode = FakeSampling
        GenerationOptions = FakeOptions

    monkeypatch.setattr(apple_evaluation.sys, "platform", "darwin")
    monkeypatch.setattr(apple_evaluation.importlib, "import_module", lambda _name: FakeSDK)
    monkeypatch.setattr(apple_evaluation.importlib.metadata, "version", lambda _name: "0.2.1")
    provider = AppleFMProvider()
    result = asyncio.run(
        provider.infer(
            instructions="local only",
            prompt="synthetic case",
            schema={"type": "object"},
            max_tokens=512,
        )
    )
    assert result["decision"] == "posted"
    assert seen["session"]["instructions"] == "local only"
    assert seen["respond"]["json_schema"] == {"type": "object"}
    assert seen["options"] == {"sampling": "greedy", "maximum_response_tokens": 512}


def test_guided_mapping_rejects_nonposted_payload_and_wrong_offset_type() -> None:
    none_case = _flat_gold(_cases()[6])
    with pytest.raises(ValueError):
        _map_guided_result({**none_case, "amount_value": "9.00"})
    with pytest.raises(ValueError):
        _map_guided_result({**none_case, "amount_start": True})
