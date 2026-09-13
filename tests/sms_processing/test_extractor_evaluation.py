"""Synthetic-only tests for the local direct-extractor evaluation workflow."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from pocketfinancer_sms import cli
from pocketfinancer_sms import evaluation as evaluation_module
from pocketfinancer_sms.analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.evaluation import (
    EvaluationInterrupted,
    ExtractorEvaluationError,
    _apply_analysis_mode,
    _canonical_gold,
    _group_metrics,
    _load_suite,
    _metrics,
    evaluate_extractor,
)
from pocketfinancer_sms.provenance import PrivateArtifactError


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/sms_processing/fixtures/extractor/synthetic-suite.jsonl"


class FakeModel:
    metadata = {"tokenizer.chat_template": "{{ messages }}"}

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("fake model received an unexpected invocation")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        if isinstance(output, dict):
            return output
        return {
            "choices": [
                {
                    "message": {"content": output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }


class FakeFactory:
    def __init__(self, outputs: list[Any], *, metadata: dict[str, str] | None = None) -> None:
        self.outputs = outputs
        self.metadata = metadata
        self.instances: list[FakeModel] = []
        self.init_kwargs: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeModel:
        model = FakeModel(list(self.outputs))
        if self.metadata is not None:
            model.metadata = self.metadata
        self.instances.append(model)
        self.init_kwargs.append(kwargs)
        return model


def _fixture_values() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _gold_outputs() -> list[str]:
    return [
        json.dumps(row["gold"], ensure_ascii=False, separators=(",", ":"))
        for row in _fixture_values()
    ]


def _allow_tmp_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pocketfinancer_sms.evaluation.require_private_output",
        lambda _root, path: path.resolve(),
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outputs: list[Any],
    *,
    output_name: str = "output",
    seed: int = 0,
    account_catalog: Path | None = None,
    metadata: dict[str, str] | None = None,
) -> tuple[dict[str, Any], FakeFactory, Path]:
    _allow_tmp_private(monkeypatch)
    gguf = tmp_path / "local.gguf"
    gguf.write_bytes(b"synthetic local gguf")
    output = tmp_path / output_name
    factory = FakeFactory(outputs, metadata=metadata)
    summary = evaluate_extractor(
        ROOT,
        gguf=gguf,
        suite="synthetic",
        output_dir=output,
        account_catalog=account_catalog,
        model_factory=factory,
        grammar_factory=lambda path: ("grammar", path.name),
        runtime_version="synthetic-runtime",
        seed=seed,
    )
    return summary, factory, output


def test_synthetic_runner_uses_gguf_chat_template_greedy_grammar_and_no_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary, factory, output = _run(monkeypatch, tmp_path, _gold_outputs())

    assert capsys.readouterr().out == ""
    assert summary["status"] == "complete"
    assert summary["total_rows"] == len(_fixture_values())
    assert summary["metrics"]["transaction"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert summary["metrics"]["strict_posted_exact_success"]["rate"] == 1.0
    assert summary["production_readiness_claim"] is False

    assert factory.init_kwargs == [
        {
            "model_path": str((tmp_path / "local.gguf").resolve()),
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "seed": 0,
            "verbose": False,
        }
    ]
    calls = factory.instances[0].calls
    assert len(calls) == len(_fixture_values())
    assert all(call["temperature"] == 0.0 for call in calls)
    assert all(call["grammar"][0] == "grammar" for call in calls)
    assert all("timeout" not in call and "deadline" not in call for call in calls)
    first_input = json.loads(calls[0]["messages"][1]["content"])
    assert first_input["contract"] == "pocketfinancer.sms-extractor-input/1"
    assert "message" in first_input

    report_path = output / "report.json"
    checkpoint_path = output / "checkpoint.json"
    report_text = report_path.read_text(encoding="utf-8")
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    assert "INR 1,250.00" not in report_text
    assert '"raw_output"' not in report_text
    assert "INR 1,250.00" in checkpoint_text
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o600

    report = json.loads(report_text)
    assert report["provenance"]["inference"]["per_row_deadline_ms"] == 0
    assert report["provenance"]["inference"]["chat_template"] == "gguf_embedded"
    assert report["provenance"]["inference"]["grammar_constrained"] is True
    assert report["capabilities"]["limitations"]
    assert report["production_readiness_claim"] is False


def test_invalid_generation_is_reason_only_in_aggregate_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "PRIVATE_PROMPT_ECHO_SHOULD_NOT_APPEAR"
    outputs = _gold_outputs()
    outputs[0] = secret

    summary, _factory, output = _run(monkeypatch, tmp_path, outputs)

    assert summary["metrics"]["reason_code_counts"] == {"extractor_malformed_json": 1}
    report_text = (output / "report.json").read_text(encoding="utf-8")
    checkpoint_text = (output / "checkpoint.json").read_text(encoding="utf-8")
    assert secret not in report_text
    assert secret in checkpoint_text
    assert '"raw_output"' not in report_text


def test_resume_skips_completed_rows_and_refuses_provenance_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary, first_factory, output = _run(monkeypatch, tmp_path, _gold_outputs())
    assert summary["status"] == "complete"
    assert len(first_factory.instances[0].calls) == len(_fixture_values())

    resumed, resumed_factory, _ = _run(
        monkeypatch,
        tmp_path,
        [],
        output_name=output.name,
    )
    assert resumed["status"] == "complete"
    assert resumed_factory.instances[0].calls == []

    with pytest.raises(
        ExtractorEvaluationError, match="checkpoint provenance does not match"
    ):
        _run(
            monkeypatch,
            tmp_path,
            [],
            output_name=output.name,
            seed=99,
        )


def test_nonempty_output_without_checkpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _allow_tmp_private(monkeypatch)
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"local")

    with pytest.raises(ExtractorEvaluationError, match="nonempty"):
        evaluate_extractor(
            ROOT,
            gguf=gguf,
            suite="synthetic",
            output_dir=output,
            model_factory=FakeFactory([]),
            grammar_factory=lambda _path: object(),
            runtime_version="fake",
        )


def test_ctrl_c_preserves_completed_rows_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _allow_tmp_private(monkeypatch)
    gguf = tmp_path / "interrupt.gguf"
    gguf.write_bytes(b"same local model")
    output = tmp_path / "interrupt-output"
    outputs = _gold_outputs()
    interrupted_factory = FakeFactory([outputs[0], KeyboardInterrupt()])

    with pytest.raises(EvaluationInterrupted) as caught:
        evaluate_extractor(
            ROOT,
            gguf=gguf,
            suite="synthetic",
            output_dir=output,
            model_factory=interrupted_factory,
            grammar_factory=lambda _path: object(),
            runtime_version="fake",
        )
    assert caught.value.completed_rows == 1
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "interrupted"
    assert len(checkpoint["records"]) == 1

    resumed_factory = FakeFactory(outputs[1:])
    summary = evaluate_extractor(
        ROOT,
        gguf=gguf,
        suite="synthetic",
        output_dir=output,
        model_factory=resumed_factory,
        grammar_factory=lambda _path: object(),
        runtime_version="fake",
    )
    assert summary["status"] == "complete"
    assert len(resumed_factory.instances[0].calls) == len(outputs) - 1


def test_missing_embedded_chat_template_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ExtractorEvaluationError, match="embedded chat template"):
        _run(monkeypatch, tmp_path, _gold_outputs(), metadata={})


def test_output_and_account_catalog_must_be_in_private_boundary(tmp_path: Path) -> None:
    gguf = tmp_path / "local.gguf"
    gguf.write_bytes(b"local")
    with pytest.raises(PrivateArtifactError, match="outside"):
        evaluate_extractor(
            ROOT,
            gguf=gguf,
            suite="synthetic",
            output_dir=tmp_path / "outside",
            model_factory=FakeFactory([]),
            grammar_factory=lambda _path: object(),
            runtime_version="fake",
        )


def test_account_catalog_contributes_resolution_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _allow_tmp_private(monkeypatch)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "accounts": [
                    {
                        "account_id": "opaque-account-1",
                        "account_type": "bank_account",
                        "aliases": ["XX1234"],
                        "owned": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary, _factory, output = _run(
        monkeypatch,
        tmp_path,
        _gold_outputs(),
        account_catalog=catalog,
    )

    resolution = summary["metrics"]["account_resolution"]
    assert resolution["eligible_count"] == 6
    assert resolution["success_rate"] == pytest.approx(1 / 3)
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))[
        "capabilities"
    ]["account_resolution"] is True


def test_grandfathered_adapter_maps_all_203_rows_and_excludes_legacy_date() -> None:
    rows, capabilities, fingerprint = _load_suite(
        ROOT, "grandfathered", primary_currency="INR"
    )

    assert len(rows) == 203
    assert len(fingerprint) == 64
    assert capabilities["legacy_date_excluded"] is True
    assert capabilities["source_spans"] is False
    assert capabilities["account_resolution"] is False
    assert any("not a fresh test" in item for item in capabilities["limitations"])
    assert any("production-quality" in item for item in capabilities["limitations"])
    assert all("date" not in (row.gold or {}) for row in rows)
    assert {row.gold["decision"] for row in rows if row.gold} == {"none", "posted"}
    assert {
        row.gold["direction"]
        for row in rows
        if row.gold and row.gold["decision"] == "posted"
    } == {"debit", "credit"}


def test_private_v2_gold_and_unlabelled_metric_capability() -> None:
    source = "INR 5.00 was debited from account XX1234."
    label = {
        "contract": "pocketfinancer.canonical-label/2",
        "decision": "posted",
        "event": {
            "amount_value": "5.00",
            "currency": "INR",
            "amount_span": {
                "start_scalar": 0,
                "end_scalar": 8,
                "text": "INR 5.00",
            },
            "direction": "debit",
            "direction_span": {
                "start_scalar": 13,
                "end_scalar": 20,
                "text": "debited",
            },
            "account_reference": "XX1234",
            "account_span": {
                "start_scalar": 34,
                "end_scalar": 40,
                "text": "XX1234",
            },
            "counterparty": None,
            "counterparty_span": None,
        },
    }
    gold = _canonical_gold(label, source)
    assert gold["minor_units"] == 500
    assert gold["account_reference"] == "1234"
    records = [
        {
            "outcome": "none",
            "gold": {"decision": "none"},
            "latency_ms": 10.0,
            "account_resolution": None,
            "reason_code": None,
        },
        {
            "outcome": "posted",
            "prediction": {},
            "gold": None,
            "latency_ms": 20.0,
            "account_resolution": None,
            "reason_code": None,
        },
    ]
    metrics = _metrics(records)
    assert metrics["labelled_count"] == 1
    assert metrics["latency_ms"]["count"] == 2
    assert metrics["transaction"]["precision"] is None


def test_private_group_metrics_suppress_small_groups() -> None:
    base = {
        "outcome": "none",
        "gold": {"decision": "none"},
        "latency_ms": 1.0,
        "account_resolution": None,
        "reason_code": None,
    }
    records = [
        {**base, "groups": {"sender_family": "small", "template_family": "small-template"}}
        for _ in range(4)
    ] + [
        {**base, "groups": {"sender_family": "large", "template_family": "large-template"}}
        for _ in range(5)
    ]

    grouped = _group_metrics(records, suite="private-canonical")

    assert set(grouped["sender_family"]["groups"]) == {"large"}
    assert grouped["sender_family"]["suppressed_group_count"] == 1
    assert grouped["sender_family"]["suppressed_row_count"] == 4
    assert set(grouped["template_family"]["groups"]) == {"large-template"}


def test_conflicting_advisory_mode_changes_hints_without_changing_source_binding() -> None:
    source = "INR 12.00 was debited from account XX1234."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india")),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    ).analyze(
        source,
        operation_id="synthetic-conflict",
        operation_config_hash="a" * 64,
        is_outgoing=False,
    )

    conflicting = _apply_analysis_mode(analysis, "conflicting")

    assert conflicting.source_fingerprint == analysis.source_fingerprint
    assert conflicting.candidates != analysis.candidates
    assert conflicting.candidates[0].evidence == analysis.candidates[0].evidence


def test_cli_emits_only_aggregate_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def fake_evaluate(repo_root: Path, **kwargs: Any) -> dict[str, Any]:
        seen["repo_root"] = repo_root
        seen.update(kwargs)
        return {
            "status": "complete",
            "suite": "synthetic",
            "completed_rows": 8,
            "total_rows": 8,
            "configuration_sha256": "a" * 64,
            "metrics": {"transaction": {"f1": 1.0}},
            "production_readiness_claim": False,
        }

    monkeypatch.setattr(cli, "evaluate_extractor", fake_evaluate)
    code = cli.main(
        [
            "--repo-root",
            str(ROOT),
            "evaluate-extractor",
            "--gguf",
            str(tmp_path / "model.gguf"),
            "--suite",
            "synthetic",
            "--output-dir",
            str(tmp_path / "private"),
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert "raw_output" not in output
    assert seen["suite"] == "synthetic"
    assert seen["enabled_profile_ids"] == ("core-en", "india")


def test_runtime_is_loaded_lazily_and_never_downloaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _allow_tmp_private(monkeypatch)
    gguf = tmp_path / "local.gguf"
    gguf.write_bytes(b"local")

    def unavailable(name: str) -> Any:
        assert name == "llama_cpp"
        raise ImportError("not installed")

    monkeypatch.setattr(evaluation_module.importlib, "import_module", unavailable)
    with pytest.raises(ExtractorEvaluationError, match="local llama-cpp-python"):
        evaluate_extractor(
            ROOT,
            gguf=gguf,
            suite="synthetic",
            output_dir=tmp_path / "output",
        )


def test_length_finish_is_aggregate_reason_and_raw_stays_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "PRIVATE_TRUNCATED_GENERATION"
    truncated = {
        "choices": [
            {
                "message": {"content": secret},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 512},
    }
    outputs = _gold_outputs()
    outputs[0] = truncated

    summary, _factory, output = _run(monkeypatch, tmp_path, outputs)

    assert summary["metrics"]["reason_code_counts"] == {
        "runtime_output_truncated": 1
    }
    assert summary["metrics"]["outcome_rates"]["runtime_failure"] == pytest.approx(
        1 / len(outputs)
    )
    assert secret not in (output / "report.json").read_text(encoding="utf-8")
    assert secret in (output / "checkpoint.json").read_text(encoding="utf-8")


def test_strict_metric_ignores_unavailable_gold_spans_and_incomplete_gold() -> None:
    amount_span = {"start_scalar": 0, "end_scalar": 8, "text": "INR 5.00"}
    posted_gold = {
        "decision": "posted",
        "minor_units": 500,
        "currency": "INR",
        "direction": "debit",
        "account_reference": "1234",
        "counterparty": None,
        "counterparty_available": False,
        "spans": {
            "amount": amount_span,
            "direction": None,
            "account": None,
            "counterparty": None,
        },
    }
    prediction = {
        **posted_gold,
        "spans": {
            "amount": amount_span,
            "direction": {"start_scalar": 13, "end_scalar": 20, "text": "debited"},
            "account": {"start_scalar": 34, "end_scalar": 40, "text": "XX1234"},
            "counterparty": None,
        },
    }
    records = [
        {
            "outcome": "posted",
            "prediction": prediction,
            "gold": posted_gold,
            "latency_ms": 10.0,
            "account_resolution": None,
            "reason_code": None,
        },
        {
            "outcome": "posted",
            "prediction": prediction,
            "gold": {**posted_gold, "account_reference": None},
            "latency_ms": 20.0,
            "account_resolution": None,
            "reason_code": None,
        },
    ]

    metrics = _metrics(records)

    assert metrics["strict_posted_exact_success"] == {
        "count": 1,
        "eligible_count": 1,
        "rate": 1.0,
    }
    assert metrics["latency_ms"] == {
        "count": 2,
        "mean": 15.0,
        "p50": 10.0,
        "p95": 20.0,
        "p99": 20.0,
        "max": 20.0,
    }


def test_cli_interruption_is_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt(_repo_root: Path, **_kwargs: Any) -> dict[str, Any]:
        raise EvaluationInterrupted(3, 9)

    monkeypatch.setattr(cli, "evaluate_extractor", interrupt)
    code = cli.main(
        [
            "--repo-root",
            str(ROOT),
            "evaluate-extractor",
            "--gguf",
            str(tmp_path / "model.gguf"),
            "--suite",
            "synthetic",
            "--output-dir",
            str(tmp_path / "private"),
        ]
    )

    assert code == 130
    assert json.loads(capsys.readouterr().out) == {
        "completed_rows": 3,
        "status": "interrupted",
        "total_rows": 9,
    }
