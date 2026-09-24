"""Local Apple Foundation Models lane over the shared SMS evaluation suite and scorer."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from .analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from .currency import CurrencyContext
from .evaluation import (
    EVALUATION_CONTRACT,
    SCORING_VERSION,
    EvaluationInterrupted,
    ExtractorEvaluationError,
    _aggregate_summary,
    _apply_analysis_mode,
    _build_report,
    _failed_record,
    _load_checkpoint,
    _load_evaluation_manifest,
    _load_suite,
    _locked_evaluation,
    _parse_record,
    _refuse_unsafe_output_reuse,
    _reject_duplicate_keys,
    _required_asset,
    _require_unique_rows,
    _resolve_user_path,
    _validate_existing_report,
)
from .extractor import build_extractor_input
from .provenance import (
    atomic_write_json,
    ensure_private_directory,
    file_sha256,
    object_sha256,
    require_private_output,
)


APPLE_PROMPT_PATH = Path("configs/sms_processing/prompts/apple-fm-sms-extractor-v1.txt")
APPLE_SCHEMA_PATH = Path("configs/sms_processing/evaluations/apple-fm-flat-v1.schema.json")
_LOCALE = re.compile(r"^[a-z]{2,3}_[A-Z]{2}$")
_FIELDS = (
    "decision",
    "amount_value",
    "amount_currency",
    "amount_start",
    "amount_end",
    "amount_text",
    "direction_value",
    "direction_start",
    "direction_end",
    "direction_text",
    "account_reference",
    "account_start",
    "account_end",
    "account_text",
    "counterparty_value",
    "counterparty_start",
    "counterparty_end",
    "counterparty_text",
)
_STRING_FIELDS = {key for key in _FIELDS if not key.endswith(("_start", "_end"))}


def _mac_system_fact(*command: str) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"
    value = result.stdout.strip()
    return value if re.fullmatch(r"[A-Za-z0-9.,_-]{1,80}", value) else "unavailable"


class AppleProvider(Protocol):
    provider_id: str
    runtime_version: str

    async def infer(
        self, *, instructions: str, prompt: str, schema: Mapping[str, Any], max_tokens: int
    ) -> Mapping[str, Any]: ...


class AppleFMProvider:
    """Official macOS SDK adapter; no import or model access occurs on WSL."""

    provider_id = "apple-fm-sdk-system-language-model-general"

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise ExtractorEvaluationError(
                "live Apple Foundation Models inference requires compatible macOS; "
                "WSL can run only fake-provider tests"
            )
        try:
            fm = importlib.import_module("apple_fm_sdk")
            model = fm.SystemLanguageModel()
            available, _reason = model.is_available()
        except (ImportError, OSError) as exc:
            raise ExtractorEvaluationError(
                "apple-fm-sdk is unavailable; install the optional macOS dependency"
            ) from exc
        except Exception as exc:
            raise ExtractorEvaluationError("Apple Foundation Models initialization failed") from exc
        if not available:
            raise ExtractorEvaluationError(
                "Apple Foundation Models is unavailable on this Mac; check Apple Intelligence, "
                "Xcode, and model download"
            )
        self.fm = fm
        self.model = model
        try:
            self.runtime_version = importlib.metadata.version("apple-fm-sdk")
        except importlib.metadata.PackageNotFoundError:
            self.runtime_version = "unknown"

    async def infer(
        self, *, instructions: str, prompt: str, schema: Mapping[str, Any], max_tokens: int
    ) -> Mapping[str, Any]:
        # A new session per SMS prevents previous cases from entering model context.
        session = self.fm.LanguageModelSession(instructions=instructions, model=self.model)
        options = self.fm.GenerationOptions(
            sampling=self.fm.SamplingMode.greedy(), maximum_response_tokens=max_tokens
        )
        generated = await session.respond(prompt, json_schema=dict(schema), options=options)
        if not generated.is_complete:
            raise ValueError("incomplete guided output")
        return json.loads(generated.to_json(), object_pairs_hook=_reject_duplicate_keys)


def evaluate_apple_foundation_models(
    repo_root: Path,
    *,
    suite: str,
    output_dir: Path,
    locale: str = "en_US",
    primary_currency: str = "INR",
    enabled_profile_ids: tuple[str, ...] = ("core-en", "india"),
    max_tokens: int = 512,
    provider: AppleProvider | None = None,
) -> dict[str, Any]:
    """Run one private local suite, returning aggregate-only metadata."""

    if provider is None and sys.platform != "darwin":
        raise ExtractorEvaluationError(
            "live Apple Foundation Models inference requires compatible macOS; "
            "WSL can run only fake-provider tests"
        )
    return _evaluate_apple_locked(
        repo_root,
        suite=suite,
        output_dir=output_dir,
        locale=locale,
        primary_currency=primary_currency,
        enabled_profile_ids=enabled_profile_ids,
        max_tokens=max_tokens,
        provider=provider,
    )


@_locked_evaluation
def _evaluate_apple_locked(
    repo_root: Path,
    *,
    suite: str,
    output_dir: Path,
    locale: str,
    primary_currency: str,
    enabled_profile_ids: tuple[str, ...],
    max_tokens: int,
    provider: AppleProvider | None,
) -> dict[str, Any]:
    if suite != "synthetic":
        raise ExtractorEvaluationError(
            "Apple Python lane currently accepts only the declared synthetic cohort"
        )
    if not _LOCALE.fullmatch(locale) or max_tokens <= 0 or not enabled_profile_ids:
        raise ExtractorEvaluationError("Apple evaluation configuration is invalid")
    root = repo_root.resolve()
    manifest, manifest_hash = _load_evaluation_manifest(root, suite)
    rows, capabilities, dataset_fingerprint = _load_suite(
        root, suite, primary_currency=primary_currency
    )
    _require_unique_rows(rows)
    prompt_path = _required_asset(root, APPLE_PROMPT_PATH, "Apple extractor instructions")
    schema_path = _required_asset(root, APPLE_SCHEMA_PATH, "Apple guided schema")
    try:
        instructions_template = prompt_path.read_text(encoding="utf-8")
        instructions = (
            instructions_template + f"\nThe person's locale is {locale}. Respond in English.\n"
        )
        schema = json.loads(
            schema_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        if set(schema["required"]) != set(_FIELDS) or set(schema["properties"]) != set(_FIELDS):
            raise ValueError("guided schema fields differ")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ExtractorEvaluationError("Apple guided schema or instructions are invalid") from exc
    active_provider = provider if provider is not None else AppleFMProvider()
    provider_id = getattr(active_provider, "provider_id", None)
    runtime_version = getattr(active_provider, "runtime_version", None)
    if (
        not isinstance(provider_id, str)
        or not provider_id
        or not isinstance(runtime_version, str)
        or not runtime_version
    ):
        raise ExtractorEvaluationError("Apple provider identity is invalid")
    simulated = provider is not None
    runtime = {
        "name": provider_id,
        "version": runtime_version,
        "model_identity_kind": "system_managed_runtime" if not simulated else "fake_provider",
        "model_identifier": (
            "SystemLanguageModel.general" if not simulated else "fake-provider-no-apple-model"
        ),
        "model_revision": None,
        "model_file_sha256": None,
        "os": platform.mac_ver()[0] if not simulated else "simulated",
        "hardware": platform.machine() if not simulated else "simulated",
        "os_build": _mac_system_fact("sw_vers", "-buildVersion") if not simulated else "simulated",
        "hardware_model": _mac_system_fact("sysctl", "-n", "hw.model")
        if not simulated
        else "simulated",
        "locale": locale,
        "locale_preflight": "not_exposed_by_python_sdk; unsupported locale errors are captured",
        "simulated": simulated,
    }
    provenance = {
        "lane": "apple_fm_python",
        "suite": suite,
        "evaluation_version": EVALUATION_CONTRACT,
        "scoring_version": SCORING_VERSION,
        "semantic_contract": manifest["semantic_contract"],
        "output_contract": manifest["lanes"]["apple_fm_python"]["output_contract"],
        "prompt_version": manifest["lanes"]["apple_fm_python"]["prompt_version"],
        "schema_version": manifest["lanes"]["apple_fm_python"]["schema_version"],
        "dataset_version": manifest["cohorts"][suite]["version"],
        "dataset_fingerprint": dataset_fingerprint,
        "runtime": runtime,
        "assets": {
            "suite_manifest_sha256": manifest_hash,
            "prompt_sha256": file_sha256(prompt_path),
            "effective_instructions_sha256": object_sha256(instructions),
            "schema_sha256": file_sha256(schema_path),
            "evaluator_sha256": file_sha256(Path(__file__)),
            "shared_scorer_sha256": file_sha256(root / "src/pocketfinancer_sms/evaluation.py"),
            "strict_parser_sha256": file_sha256(root / "src/pocketfinancer_sms/extractor.py"),
            "strict_extractor_schema_sha256": file_sha256(
                root / "configs/sms_processing/contracts/v3/sms-extractor.schema.json"
            ),
        },
        "inference": {
            "max_tokens": max_tokens,
            "sampling": "greedy",
            "guided_generation": True,
            "primary_currency": primary_currency,
            "enabled_profile_ids": list(enabled_profile_ids),
            "case_context": "new_session_per_case",
            "cumulative_guided_snapshots": "unavailable_in_python_sdk",
            "decoded_tokens": "unavailable",
            "logits": "unavailable",
            "confidence": "unavailable",
            "context_size": "unavailable",
            "token_throughput": "unavailable",
        },
    }
    configuration_hash = object_sha256(provenance)
    private_output = require_private_output(root, _resolve_user_path(root, output_dir))
    ensure_private_directory(private_output)
    _refuse_unsafe_output_reuse(private_output)
    checkpoint_path = private_output / "checkpoint.json"
    completed = _load_checkpoint(
        checkpoint_path,
        configuration_hash=configuration_hash,
        dataset_fingerprint=dataset_fingerprint,
        row_ids={row.row_id for row in rows},
    )
    _validate_existing_report(
        private_output / "report.json",
        configuration_hash=configuration_hash,
        dataset_fingerprint=dataset_fingerprint,
    )
    analyzer = DeterministicSmsAnalyzer(
        CurrencyContext(primary_currency, enabled_profile_ids),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    )
    checkpoint = {
        "contract": EVALUATION_CONTRACT,
        "status": "running",
        "configuration_sha256": configuration_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "total_rows": len(rows),
        "records": completed,
    }
    atomic_write_json(checkpoint_path, checkpoint)
    try:
        for row in rows:
            if row.row_id in completed:
                continue
            analysis = row.analysis
            if analysis is None or analysis.contract != ANALYSIS_CONTRACT_V2:
                analysis = analyzer.analyze(
                    row.source,
                    operation_id=f"evaluate-{row.row_id}",
                    is_outgoing=False,
                    operation_config_hash=configuration_hash,
                )
            analysis = _apply_analysis_mode(analysis, row.analysis_mode)
            extractor_input = build_extractor_input(
                row.source,
                analysis,
                sender_family=row.sender_family,
                primary_currency=primary_currency,
                enabled_profile_ids=enabled_profile_ids,
            )
            request = json.dumps(extractor_input, ensure_ascii=False, separators=(",", ":"))
            started = time.perf_counter()
            try:
                flat = asyncio.run(
                    active_provider.infer(
                        instructions=instructions,
                        prompt=request,
                        schema=schema,
                        max_tokens=max_tokens,
                    )
                )
                mapped = _map_guided_result(flat)
                raw = json.dumps(mapped, ensure_ascii=False, separators=(",", ":"))
                record = _parse_record(
                    row,
                    raw,
                    (time.perf_counter() - started) * 1000.0,
                    primary_currency=primary_currency,
                    enabled_profile_ids=enabled_profile_ids,
                    catalog=None,
                    usage={},
                )
                record["guided_output"] = dict(flat)
                record["validation_stages"] = {
                    "guided_schema": "complete",
                    "mapping": "complete",
                    "strict_parser": "accepted" if record["reason_code"] is None else "rejected",
                }
            except KeyboardInterrupt:
                raise
            except (TypeError, ValueError) as exc:
                record = _failed_record(
                    row,
                    (time.perf_counter() - started) * 1000.0,
                    "malformed",
                    _apple_error_code(exc),
                )
            except Exception as exc:
                record = _failed_record(
                    row,
                    (time.perf_counter() - started) * 1000.0,
                    "runtime_failure",
                    _apple_error_code(exc),
                )
            completed[row.row_id] = record
            checkpoint["records"] = completed
            atomic_write_json(checkpoint_path, checkpoint)
    except KeyboardInterrupt as exc:
        checkpoint["status"] = "interrupted"
        atomic_write_json(checkpoint_path, checkpoint)
        report = _build_report(
            suite=suite,
            provenance=provenance,
            configuration_hash=configuration_hash,
            capabilities=capabilities,
            rows=rows,
            records=completed,
            status="interrupted",
        )
        atomic_write_json(private_output / "report.json", report)
        raise EvaluationInterrupted(len(completed), len(rows)) from exc
    checkpoint["status"] = "complete"
    atomic_write_json(checkpoint_path, checkpoint)
    report = _build_report(
        suite=suite,
        provenance=provenance,
        configuration_hash=configuration_hash,
        capabilities=capabilities,
        rows=rows,
        records=completed,
        status="complete",
    )
    atomic_write_json(private_output / "report.json", report)
    return _aggregate_summary(report)


def _map_guided_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_FIELDS):
        raise ValueError("Apple guided response fields are invalid")
    for key in _FIELDS:
        item = value[key]
        if key in _STRING_FIELDS:
            if not isinstance(item, str):
                raise ValueError("Apple guided response field type is invalid")
        elif isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("Apple guided response offset type is invalid")
    decision = value["decision"]
    if decision not in {"none", "abstain", "posted"}:
        raise ValueError("Apple guided response decision is invalid")
    if decision != "posted":
        if any(value[key] != ("" if key in _STRING_FIELDS else 0) for key in _FIELDS[1:]):
            raise ValueError("Apple non-posted response has extraction fields")
        return {"decision": decision}

    def evidence(prefix: str) -> dict[str, Any]:
        return {
            "start_scalar": value[f"{prefix}_start"],
            "end_scalar": value[f"{prefix}_end"],
            "text": value[f"{prefix}_text"],
        }

    counterparty_empty = (
        value["counterparty_value"] == ""
        and value["counterparty_text"] == ""
        and value["counterparty_start"] == 0
        and value["counterparty_end"] == 0
    )
    return {
        "decision": "posted",
        "amount": {
            "value": value["amount_value"],
            "currency": value["amount_currency"],
            "evidence": evidence("amount"),
        },
        "direction": {"value": value["direction_value"], "evidence": evidence("direction")},
        "account": {"reference": value["account_reference"], "evidence": evidence("account")},
        "counterparty": (
            None
            if counterparty_empty
            else {"value": value["counterparty_value"], "evidence": evidence("counterparty")}
        ),
    }


def _apple_error_code(exc: Exception) -> str:
    """Avoid exception messages, which may echo private prompts or SMS."""

    known = {
        "UnsupportedLanguageOrLocaleError": "apple_unsupported_locale",
        "GuardrailViolationError": "apple_guardrail_violation",
        "RefusalError": "apple_refusal",
        "ExceededContextWindowSizeError": "apple_context_exceeded",
        "AssetsUnavailableError": "apple_assets_unavailable",
        "InvalidGenerationSchemaError": "apple_guided_schema_incompatible",
        "DecodingFailureError": "apple_decoding_failure",
    }
    return known.get(
        type(exc).__name__,
        "apple_guided_result_invalid"
        if isinstance(exc, (TypeError, ValueError))
        else "apple_runtime_failure",
    )
