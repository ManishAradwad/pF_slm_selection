"""Private-safe, resumable local GGUF evaluation for the direct SMS extractor."""

from __future__ import annotations

import importlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .account_resolution import AccountCatalogEntry, resolve_account
from .analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from .corpus.grouping import sender_family
from .currency import CurrencyContext, ISO_MINOR_UNITS, MAX_SIGNED_64
from .extractor import (
    ExtractionValidationError,
    build_extractor_input,
    normalize_account_reference,
    normalize_counterparty,
    parse_and_normalize_extraction,
)
from .provenance import (
    PrivateArtifactError,
    atomic_write_json,
    ensure_private_directory,
    file_sha256,
    object_sha256,
    require_private_output,
)
from .types import Analysis, CandidateKind


EVALUATION_CONTRACT = "pocketfinancer.sms-extractor-evaluation/1"
PROMPT_PATH = Path("configs/sms_processing/prompts/sms-extractor-v1.txt")
GRAMMAR_PATH = Path("configs/sms_processing/grammars/sms-extractor-v1.gbnf")
SCHEMA_PATH = Path("configs/sms_processing/contracts/v3/sms-extractor.schema.json")
PROFILE_PATH = Path("configs/sms_processing/contracts/v3/extractor-validation-profile.json")
SYNTHETIC_PATH = Path("tests/sms_processing/fixtures/extractor/synthetic-suite.jsonl")
GRANDFATHERED_PATH = Path("DATA/extraction_ds.jsonl")
_SAFE_REASON = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_SMALL_GROUP_MINIMUM = 5


class ExtractorEvaluationError(PrivateArtifactError):
    """A fail-closed evaluation error whose message contains no row data."""


class EvaluationInterrupted(ExtractorEvaluationError):
    """The user interrupted evaluation after completed rows were checkpointed."""

    def __init__(self, completed_rows: int, total_rows: int) -> None:
        self.completed_rows = completed_rows
        self.total_rows = total_rows
        super().__init__("extractor evaluation interrupted; completed rows were preserved")


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    row_id: str
    source: str
    sender_family: str
    gold: Mapping[str, Any] | None
    analysis: Analysis | None
    analysis_mode: str
    groups: Mapping[str, str]
    fingerprint_binding: Mapping[str, Any]


def evaluate_extractor(
    repo_root: Path,
    *,
    gguf: Path,
    suite: str,
    output_dir: Path,
    account_catalog: Path | None = None,
    primary_currency: str = "INR",
    enabled_profile_ids: Sequence[str] = ("core-en", "india"),
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    seed: int = 0,
    max_tokens: int = 512,
    model_factory: Callable[..., Any] | None = None,
    grammar_factory: Callable[[Path], Any] | None = None,
    runtime_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate one local GGUF and return only aggregate-safe result metadata."""

    root = repo_root.resolve()
    if suite not in {"synthetic", "grandfathered", "private-canonical"}:
        raise ExtractorEvaluationError("extractor evaluation suite is unsupported")
    if n_ctx <= 0 or max_tokens <= 0 or not enabled_profile_ids:
        raise ExtractorEvaluationError("extractor evaluation configuration is invalid")
    model_path = gguf.expanduser().resolve()
    if not model_path.is_file():
        raise ExtractorEvaluationError("local GGUF is unavailable")
    private_output = require_private_output(root, _resolve_user_path(root, output_dir))
    ensure_private_directory(private_output)
    _refuse_unsafe_output_reuse(private_output)

    prompt_path = _required_asset(root, PROMPT_PATH, "extractor prompt")
    grammar_path = _required_asset(root, GRAMMAR_PATH, "extractor grammar")
    schema_path = _required_asset(root, SCHEMA_PATH, "extractor schema")
    profile_path = _required_asset(root, PROFILE_PATH, "extractor validation profile")
    prompt = _read_text(prompt_path, "extractor prompt")
    rows, capabilities, dataset_fingerprint = _load_suite(
        root, suite, primary_currency=primary_currency
    )
    catalog, catalog_hash = _load_account_catalog(root, account_catalog)

    llama_module = None
    if model_factory is None or grammar_factory is None or runtime_version is None:
        try:
            llama_module = importlib.import_module("llama_cpp")
        except ImportError as exc:
            raise ExtractorEvaluationError(
                "local llama-cpp-python runtime is unavailable"
            ) from exc
    if runtime_version is None:
        runtime_version = str(getattr(llama_module, "__version__", "unknown"))
    assets = {
        "gguf_sha256": file_sha256(model_path),
        "prompt_sha256": file_sha256(prompt_path),
        "schema_sha256": file_sha256(schema_path),
        "grammar_sha256": file_sha256(grammar_path),
        "validation_profile_sha256": file_sha256(profile_path),
        "account_catalog_sha256": catalog_hash,
    }
    inference = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "greedy_decoding": True,
        "chat_template": "gguf_embedded",
        "grammar_constrained": True,
        "per_row_deadline_ms": 0,
        "primary_currency": primary_currency,
        "enabled_profile_ids": list(enabled_profile_ids),
    }
    provenance = {
        "suite": suite,
        "dataset_fingerprint": dataset_fingerprint,
        "runtime": {"name": "llama-cpp-python", "version": runtime_version},
        "assets": assets,
        "inference": inference,
    }
    configuration_hash = object_sha256(provenance)
    checkpoint_path = private_output / "checkpoint.json"
    completed = _load_checkpoint(
        checkpoint_path,
        configuration_hash=configuration_hash,
        dataset_fingerprint=dataset_fingerprint,
        row_ids={row.row_id for row in rows},
    )

    if grammar_factory is None:
        grammar_class = getattr(llama_module, "LlamaGrammar")

        def load_grammar(path: Path) -> Any:
            return grammar_class.from_file(str(path))

        grammar_factory = load_grammar
    if model_factory is None:
        model_factory = getattr(llama_module, "Llama")
    try:
        grammar = grammar_factory(grammar_path)
        model = model_factory(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
    except Exception as exc:
        raise ExtractorEvaluationError("local GGUF runtime initialization failed") from exc
    _require_embedded_chat_template(model)

    analyzer = DeterministicSmsAnalyzer(
        CurrencyContext(primary_currency, tuple(enabled_profile_ids)),
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
                enabled_profile_ids=tuple(enabled_profile_ids),
            )
            started = time.perf_counter()
            try:
                response = model.create_chat_completion(
                    messages=(
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                extractor_input,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    grammar=grammar,
                    seed=seed,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                raw_output, finish_reason, usage = _chat_response(response)
                if finish_reason == "length":
                    result = _failed_record(
                        row,
                        latency_ms,
                        "runtime_failure",
                        "runtime_output_truncated",
                        raw_output=raw_output,
                        usage=usage,
                    )
                else:
                    result = _parse_record(
                        row,
                        raw_output,
                        latency_ms,
                        primary_currency=primary_currency,
                        enabled_profile_ids=enabled_profile_ids,
                        catalog=catalog,
                        usage=usage,
                    )
            except KeyboardInterrupt:
                raise
            except Exception:
                latency_ms = (time.perf_counter() - started) * 1000.0
                result = _failed_record(
                    row,
                    latency_ms,
                    "runtime_failure",
                    "runtime_failure",
                )
            completed[row.row_id] = result
            checkpoint["records"] = completed
            atomic_write_json(checkpoint_path, checkpoint)
    except KeyboardInterrupt as exc:
        checkpoint["status"] = "interrupted"
        checkpoint["records"] = completed
        atomic_write_json(checkpoint_path, checkpoint)
        partial_report = _build_report(
            suite=suite,
            provenance=provenance,
            configuration_hash=configuration_hash,
            capabilities=capabilities,
            rows=rows,
            records=completed,
            status="interrupted",
        )
        atomic_write_json(private_output / "report.json", partial_report)
        raise EvaluationInterrupted(len(completed), len(rows)) from exc
    finally:
        del model

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


def _resolve_user_path(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return (root / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


def _required_asset(root: Path, relative: Path, label: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise ExtractorEvaluationError(f"{label} is unavailable")
    return path


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExtractorEvaluationError(f"{label} could not be read") from exc


def _refuse_unsafe_output_reuse(output_dir: Path) -> None:
    allowed = {"checkpoint.json", "report.json"}
    if any(path.name not in allowed for path in output_dir.iterdir()):
        raise ExtractorEvaluationError("nonempty evaluation output cannot be reused")
    if (output_dir / "report.json").exists() and not (output_dir / "checkpoint.json").exists():
        raise ExtractorEvaluationError("nonempty evaluation output cannot be reused")


def _load_checkpoint(
    path: Path,
    *,
    configuration_hash: str,
    dataset_fingerprint: str,
    row_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractorEvaluationError("evaluation checkpoint is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("contract") != EVALUATION_CONTRACT
        or value.get("configuration_sha256") != configuration_hash
        or value.get("dataset_fingerprint") != dataset_fingerprint
        or value.get("total_rows") != len(row_ids)
        or not isinstance(value.get("records"), dict)
        or not set(value["records"]) <= row_ids
    ):
        raise ExtractorEvaluationError("evaluation checkpoint provenance does not match")
    return dict(value["records"])


def _load_suite(
    root: Path,
    suite: str,
    *,
    primary_currency: str,
) -> tuple[list[EvaluationRow], dict[str, Any], str]:
    if suite == "synthetic":
        path = _required_asset(root, SYNTHETIC_PATH, "synthetic extractor suite")
        rows = _load_synthetic(path, primary_currency=primary_currency)
        capabilities = {
            "human_gold": False,
            "decision": True,
            "normalized_fields": True,
            "source_spans": True,
            "account_resolution": False,
            "transaction_timestamp": False,
            "limitations": [
                "sanitized synthetic coverage is not production-quality evidence",
                "synthetic grouping labels do not estimate private sender prevalence",
            ],
        }
        fingerprint = object_sha256(
            {"asset_sha256": file_sha256(path), "rows": [row.fingerprint_binding for row in rows]}
        )
        return rows, capabilities, fingerprint
    if suite == "grandfathered":
        path = _required_asset(root, GRANDFATHERED_PATH, "grandfathered extractor suite")
        rows = _load_grandfathered(path, primary_currency=primary_currency)
        if len(rows) != 203:
            raise ExtractorEvaluationError("grandfathered extractor suite must contain 203 rows")
        capabilities = {
            "human_gold": False,
            "decision": True,
            "normalized_fields": True,
            "source_spans": False,
            "account_resolution": False,
            "transaction_timestamp": False,
            "legacy_date_excluded": True,
            "limitations": [
                "the grandfathered 203-row fixture is a regression set, not a fresh test",
                "legacy date labels are deliberately excluded",
                "gold spans, receipt-time provenance, and account-resolution truth are unavailable",
                "this suite cannot support a production-quality claim",
            ],
        }
        fingerprint = object_sha256(
            {"asset_sha256": file_sha256(path), "rows": [row.fingerprint_binding for row in rows]}
        )
        return rows, capabilities, fingerprint
    rows, private_binding, private_capabilities = _load_private_canonical(root)
    return rows, private_capabilities, object_sha256(private_binding)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractorEvaluationError(f"{label} is invalid") from exc
    if not all(isinstance(value, dict) for value in values):
        raise ExtractorEvaluationError(f"{label} is invalid")
    return values


def _load_synthetic(path: Path, *, primary_currency: str) -> list[EvaluationRow]:
    values = _read_jsonl(path, "synthetic extractor suite")
    rows: list[EvaluationRow] = []
    for value in values:
        try:
            source = value["source"]
            sender = value["sender"]
            row_id = value["id"]
            gold = _direct_gold(value["gold"], source, primary_currency)
            mode = value.get("analysis_mode", "normal")
            groups = {
                "sender_family": value["sender_family_label"],
                "template_family": value["template_family_label"],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtractorEvaluationError("synthetic extractor suite is invalid") from exc
        if not all(isinstance(item, str) and item for item in (source, sender, row_id)):
            raise ExtractorEvaluationError("synthetic extractor suite is invalid")
        if mode not in {"normal", "zero", "incomplete", "conflicting"}:
            raise ExtractorEvaluationError("synthetic extractor suite is invalid")
        rows.append(
            EvaluationRow(
                row_id=row_id,
                source=source,
                sender_family=sender_family(sender),
                gold=gold,
                analysis=None,
                analysis_mode=mode,
                groups=groups,
                fingerprint_binding={
                    "id": row_id,
                    "source_sha256": object_sha256(source),
                    "gold_sha256": object_sha256(gold),
                    "analysis_mode": mode,
                    "groups": groups,
                },
            )
        )
    if len({row.row_id for row in rows}) != len(rows):
        raise ExtractorEvaluationError("synthetic extractor suite has duplicate row IDs")
    return rows


def _load_grandfathered(path: Path, *, primary_currency: str) -> list[EvaluationRow]:
    values = _read_jsonl(path, "grandfathered extractor suite")
    rows: list[EvaluationRow] = []
    for index, value in enumerate(values):
        try:
            source = value["sms"]
            sender = value["sender"]
            legacy = json.loads(value["expected"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ExtractorEvaluationError("grandfathered extractor suite is invalid") from exc
        if not isinstance(source, str) or not isinstance(sender, str):
            raise ExtractorEvaluationError("grandfathered extractor suite is invalid")
        if legacy is None:
            gold = {"decision": "none"}
        elif isinstance(legacy, dict) and legacy.get("type") in {"debit", "credit"}:
            try:
                minor_units = _minor_units(legacy["amount"], primary_currency)
            except (KeyError, ValueError) as exc:
                raise ExtractorEvaluationError("grandfathered extractor suite is invalid") from exc
            gold = {
                "decision": "posted",
                "minor_units": minor_units,
                "currency": primary_currency,
                "direction": legacy["type"],
                "account_reference": _account_compare(legacy.get("account")),
                "counterparty": _counterparty_compare(legacy.get("counterparty")),
                "counterparty_available": legacy.get("counterparty") is not None,
                "spans": None,
            }
        else:
            raise ExtractorEvaluationError("grandfathered extractor suite is invalid")
        row_id = f"grandfathered-{index + 1:03d}"
        rows.append(
            EvaluationRow(
                row_id=row_id,
                source=source,
                sender_family=sender_family(sender),
                gold=gold,
                analysis=None,
                analysis_mode="normal",
                groups={},
                fingerprint_binding={
                    "id": row_id,
                    "source_sha256": object_sha256(source),
                    "gold_sha256": object_sha256(gold),
                },
            )
        )
    return rows


def _load_private_canonical(
    root: Path,
) -> tuple[list[EvaluationRow], dict[str, Any], dict[str, Any]]:
    private_root = require_private_output(root, root / "PRIVATE_DATA/sms_processing")
    current_path = private_root / "CURRENT.json"
    database_path = private_root / "workbench/workbench-v2.sqlite3"
    if not current_path.is_file() or not database_path.is_file():
        raise ExtractorEvaluationError(
            "private-canonical suite requires the secure workbench and canonical corpus"
        )
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
        run_id = current["run_id"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExtractorEvaluationError("private canonical corpus pointer is invalid") from exc
    from .workbench.secure_store import KeyringKeyProvider, SecureWorkbenchStore

    try:
        store = SecureWorkbenchStore(database_path, key_provider=KeyringKeyProvider())
        store.verify_revision_chains()
        listed = store.list_rows(
            filters={}, search=None, sort="source_id", descending=False, limit=200, offset=0
        )
        summaries = list(listed["rows"])
        while len(summaries) < int(listed["total"]):
            page = store.list_rows(
                filters={},
                search=None,
                sort="source_id",
                descending=False,
                limit=200,
                offset=len(summaries),
            )
            summaries.extend(page["rows"])
    except Exception as exc:
        raise ExtractorEvaluationError("secure private workbench could not be opened") from exc
    rows: list[EvaluationRow] = []
    binding_rows: list[dict[str, Any]] = []
    gold_count = 0
    span_gold_count = 0
    for summary in summaries:
        source_id = str(summary["source_id"])
        record = store.get_record(source_id)
        if record is None:
            raise ExtractorEvaluationError("secure private workbench is inconsistent")
        annotations = store.submitted_annotations(source_id)
        selected = _select_committed_annotation(annotations)
        gold = None
        revision_hash = None
        if selected is not None and selected.get("canonical_label") is not None:
            gold = _canonical_gold(selected["canonical_label"], record["source"]["body"])
            revision_hash = selected["revision_hash"]
            gold_count += 1
            span_gold_count += int(gold.get("spans") is not None)
        grouping = record["grouping"]
        groups = {
            "sender_family": grouping["sender_family_hash"],
            "template_family": grouping["normalized_template_hash"],
        }
        row = EvaluationRow(
            row_id=source_id,
            source=record["source"]["body"],
            sender_family=sender_family(record["source"].get("sender", "")),
            gold=gold,
            analysis=Analysis.from_dict(record["analysis"], source=record["source"]["body"]),
            analysis_mode="normal",
            groups=groups,
            fingerprint_binding={
                "source_id": source_id,
                "record_fingerprint": record["analysis"]["source_fingerprint"],
                "revision_hash": revision_hash,
                "groups": groups,
            },
        )
        rows.append(row)
        binding_rows.append(dict(row.fingerprint_binding))
    if not rows:
        raise ExtractorEvaluationError("private canonical corpus contains no rows")
    capabilities = {
        "human_gold": True,
        "decision": gold_count > 0,
        "normalized_fields": gold_count > 0,
        "source_spans": span_gold_count > 0,
        "account_resolution": False,
        "transaction_timestamp": False,
        "submitted_or_adjudicated_gold_rows": gold_count,
        "unlabelled_outcome_latency_only_rows": len(rows) - gold_count,
        "limitations": [
            "accuracy includes submitted/adjudicated human-gold rows only",
            "unlabelled rows contribute only outcome and latency rates",
            "private sender/template groups below the suppression threshold are hidden",
            "native receipt-time and Apple Foundation Models behavior require separate evaluation",
        ],
    }
    return rows, {"corpus_run_id": run_id, "rows": binding_rows}, capabilities


def _select_committed_annotation(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    adjudicated = [value for value in values if value.get("status") == "adjudicated"]
    if adjudicated:
        return max(adjudicated, key=lambda value: int(value.get("revision", 0)))
    submitted = [value for value in values if value.get("status") == "submitted"]
    if len(submitted) == 1:
        return submitted[0]
    if submitted and len(
        {object_sha256(value.get("canonical_label")) for value in submitted}
    ) == 1:
        return submitted[0]
    return None


def _direct_gold(value: Any, source: str, primary_currency: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("decision") not in {"none", "abstain", "posted"}
    ):
        raise ValueError("invalid direct gold")
    if value["decision"] != "posted":
        return {"decision": value["decision"]}
    amount = value["amount"]
    direction = value["direction"]
    account = value["account"]
    counterparty = value.get("counterparty")
    currency = amount.get("currency", primary_currency)
    spans = {
        "amount": _checked_span(amount["evidence"], source),
        "direction": _checked_span(direction["evidence"], source),
        "account": _checked_span(account["evidence"], source),
        "counterparty": (
            _checked_span(counterparty["evidence"], source)
            if counterparty is not None
            else None
        ),
    }
    return {
        "decision": "posted",
        "minor_units": _minor_units(amount["value"], currency),
        "currency": currency,
        "direction": direction["value"],
        "account_reference": _account_compare(account["reference"]),
        "counterparty": (
            _counterparty_compare(counterparty["value"]) if counterparty is not None else None
        ),
        "counterparty_available": counterparty is not None,
        "spans": spans,
    }


def _canonical_gold(value: Mapping[str, Any], source: str) -> dict[str, Any]:
    if value.get("contract") == "pocketfinancer.canonical-label/2":
        decision = str(value.get("decision", ""))
        if decision in {"none", "abstain"}:
            return {"decision": decision}
        event = value.get("event")
        if decision != "posted" or not isinstance(event, Mapping):
            raise ExtractorEvaluationError("private canonical label is unsupported")
        direction_span = event.get("direction_span")
        return {
            "decision": "posted",
            "minor_units": _minor_units(event["amount_value"], event["currency"]),
            "currency": event["currency"],
            "direction": event["direction"],
            "account_reference": _account_compare(event["account_reference"]),
            "counterparty": _counterparty_compare(event.get("counterparty")),
            "counterparty_available": event.get("counterparty") is not None,
            "spans": {
                "amount": _checked_span(event["amount_span"], source),
                "direction": (
                    _checked_span(direction_span, source)
                    if direction_span is not None
                    else None
                ),
                "account": _checked_span(event["account_span"], source),
                "counterparty": (
                    _checked_span(event["counterparty_span"], source)
                    if event.get("counterparty_span") is not None
                    else None
                ),
            },
        }
    decision = str(value.get("decision", ""))
    if decision in {"not_posted", "non_financial", "none"}:
        return {"decision": "none"}
    if decision in {"ambiguous", "multiple_event", "abstain"}:
        return {"decision": "abstain"}
    if decision != "posted" or not value.get("events"):
        raise ExtractorEvaluationError("private canonical label is unsupported")
    event = value["events"][0]
    amount = event.get("amount") if isinstance(event.get("amount"), dict) else {}
    direction_value = event.get("direction")
    direction = direction_value if isinstance(direction_value, dict) else {}
    account = event.get("account") if isinstance(event.get("account"), dict) else {}
    counterparty_value = event.get("counterparty")
    counterparty = counterparty_value if isinstance(counterparty_value, dict) else {}
    amount_span = event.get("amount_span") or amount.get("evidence")
    direction_span = event.get("direction_span") or direction.get("evidence")
    account_span = event.get("account_span") or account.get("evidence")
    counterparty_span = event.get("counterparty_span") or counterparty.get("evidence")
    spans = None
    if amount_span and direction_span and account_span:
        spans = {
            "amount": _checked_span(amount_span, source),
            "direction": _checked_span(direction_span, source),
            "account": _checked_span(account_span, source),
            "counterparty": (
                _checked_span(counterparty_span, source) if counterparty_span else None
            ),
        }
    currency = event.get("currency") or amount.get("currency")
    minor_units = event.get("minor_units")
    if minor_units is None and amount.get("value") is not None and currency:
        minor_units = _minor_units(amount["value"], currency)
    direction_scalar = direction.get("value") if direction else direction_value
    account_reference = event.get("account_reference") or account.get("reference")
    counterparty_scalar = event.get("counterparty_value") or counterparty.get("value")
    return {
        "decision": "posted",
        "minor_units": minor_units,
        "currency": currency,
        "direction": direction_scalar,
        "account_reference": _account_compare(account_reference),
        "counterparty": _counterparty_compare(counterparty_scalar),
        "counterparty_available": counterparty_scalar is not None,
        "spans": spans,
    }


def _checked_span(value: Mapping[str, Any], source: str) -> dict[str, Any]:
    start = value.get("start_scalar", value.get("start_char"))
    end = value.get("end_scalar", value.get("end_char"))
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or end > len(source)
    ):
        raise ValueError("invalid gold span")
    text = value.get("text", source[start:end])
    if text != source[start:end]:
        raise ValueError("invalid gold span")
    return {"start_scalar": start, "end_scalar": end, "text": text}


def _minor_units(value: Any, currency: str) -> int:
    if currency not in ISO_MINOR_UNITS:
        raise ValueError("unsupported gold currency")
    decimal = Decimal(str(value))
    scaled = decimal * (10 ** ISO_MINOR_UNITS[currency])
    if (
        not scaled.is_finite()
        or scaled <= 0
        or scaled != scaled.to_integral_value()
        or scaled > MAX_SIGNED_64
    ):
        raise ValueError("invalid gold precision")
    return int(scaled)


def _apply_analysis_mode(analysis: Analysis, mode: str) -> Analysis:
    if mode == "normal":
        return analysis
    if mode == "zero":
        return replace(analysis, clauses=(), candidates=(), cues=(), reason_codes=())
    if mode == "incomplete":
        candidates = tuple(
            candidate
            for candidate in analysis.candidates
            if candidate.kind == CandidateKind.AMOUNT
        )
        return replace(analysis, candidates=candidates, cues=())
    if mode == "conflicting":
        candidates = []
        for candidate in analysis.candidates:
            replacement = dict(candidate.value)
            if candidate.kind == CandidateKind.DIRECTION:
                direction = replacement.get("direction")
                replacement["direction"] = (
                    "credit" if direction == "debit" else "debit"
                )
            elif candidate.kind == CandidateKind.ACCOUNT:
                replacement["normalized_identifier"] = "0000"
            elif candidate.kind == CandidateKind.AMOUNT and isinstance(
                replacement.get("minor_units"), int
            ):
                replacement["minor_units"] += 1
            candidates.append(replace(candidate, value=replacement))
        return replace(analysis, candidates=tuple(candidates))
    raise ExtractorEvaluationError("synthetic analysis mode is invalid")


def _require_embedded_chat_template(model: Any) -> None:
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, Mapping) or not any(
        str(key).startswith("tokenizer.chat_template") and value
        for key, value in metadata.items()
    ):
        raise ExtractorEvaluationError("GGUF does not provide an embedded chat template")


def _chat_response(response: Any) -> tuple[str, str, dict[str, int]]:
    try:
        choice = response["choices"][0]
        raw = choice["message"]["content"]
        finish = str(choice.get("finish_reason") or "unknown")
        usage_value = response.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_value.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_value.get("completion_tokens", 0)),
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid local runtime response") from exc
    if not isinstance(raw, str):
        raise RuntimeError("invalid local runtime response")
    return raw, finish, usage


def _parse_record(
    row: EvaluationRow,
    raw_output: str,
    latency_ms: float,
    *,
    primary_currency: str,
    enabled_profile_ids: Sequence[str],
    catalog: Sequence[AccountCatalogEntry] | None,
    usage: Mapping[str, int],
) -> dict[str, Any]:
    try:
        parsed = parse_and_normalize_extraction(
            raw_output,
            row.source,
            primary_currency=primary_currency,
            enabled_profile_ids=tuple(enabled_profile_ids),
        )
        prediction = _prediction_view(parsed)
    except ExtractionValidationError as exc:
        reason_code = _safe_reason_code(exc)
        outcome = "runtime_failure" if reason_code.startswith("runtime_") else "malformed"
        return _failed_record(
            row,
            latency_ms,
            outcome,
            reason_code,
            raw_output=raw_output,
            usage=usage,
        )
    except Exception:
        return _failed_record(
            row,
            latency_ms,
            "runtime_failure",
            "runtime_failure",
            usage=usage,
        )
    resolution = (
        _resolve_for_evaluation(prediction.get("account_reference"), catalog)
        if prediction["decision"] == "posted"
        else None
    )
    return {
        "row_id": row.row_id,
        "outcome": prediction["decision"],
        "reason_code": None,
        "latency_ms": latency_ms,
        "prediction": prediction,
        "gold": dict(row.gold) if row.gold is not None else None,
        "groups": dict(row.groups),
        "account_resolution": resolution,
        "usage": dict(usage),
        "raw_output": raw_output,
    }


def _failed_record(
    row: EvaluationRow,
    latency_ms: float,
    outcome: str,
    reason_code: str,
    *,
    raw_output: str | None = None,
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "outcome": outcome,
        "reason_code": reason_code,
        "latency_ms": latency_ms,
        "prediction": None,
        "gold": dict(row.gold) if row.gold is not None else None,
        "groups": dict(row.groups),
        "account_resolution": None,
        "usage": dict(usage or {}),
        "raw_output": raw_output,
    }


def _safe_reason_code(exc: Exception) -> str:
    candidate = getattr(exc, "reason_code", None)
    if candidate is None and exc.args:
        candidate = exc.args[0]
    if isinstance(candidate, str) and _SAFE_REASON.fullmatch(candidate):
        return candidate
    return "extractor_output_invalid"


def _prediction_view(value: Any) -> dict[str, Any]:
    decision = getattr(value, "decision", None)
    transaction = getattr(value, "transaction", None)
    if isinstance(value, Mapping):
        decision = value.get("decision")
        transaction = value.get("transaction")
    if hasattr(decision, "value"):
        decision = decision.value
    if decision not in {"none", "abstain", "posted"}:
        raise ValueError("extractor_result_invalid")
    if decision != "posted":
        return {"decision": decision}
    if transaction is None:
        raise ValueError("extractor_result_invalid")
    tx = _as_mapping(transaction)
    direction = tx.get("direction")
    if hasattr(direction, "value"):
        direction = direction.value
    return {
        "decision": "posted",
        "minor_units": tx.get("minor_units"),
        "currency": tx.get("currency"),
        "direction": direction,
        "account_reference": _account_compare(tx.get("account_reference")),
        "counterparty": _counterparty_compare(tx.get("counterparty")),
        "spans": {
            "amount": _span_view(tx.get("amount_span")),
            "direction": _span_view(tx.get("direction_span")),
            "account": _span_view(tx.get("account_span")),
            "counterparty": _span_view(tx.get("counterparty_span")),
        },
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    names = (
        "minor_units",
        "currency",
        "direction",
        "account_reference",
        "counterparty",
        "amount_span",
        "direction_span",
        "account_span",
        "counterparty_span",
    )
    return {name: getattr(value, name, None) for name in names}


def _span_view(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    mapping = _as_mapping(value)
    return {
        "start_scalar": mapping.get("start_scalar"),
        "end_scalar": mapping.get("end_scalar"),
        "text": mapping.get("text"),
    }


def _load_account_catalog(
    root: Path, path: Path | None
) -> tuple[list[AccountCatalogEntry] | None, str | None]:
    if path is None:
        return None, None
    resolved = require_private_output(root, _resolve_user_path(root, path))
    if not resolved.is_file():
        raise ExtractorEvaluationError("local account catalog is unavailable")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractorEvaluationError("local account catalog is invalid") from exc
    accounts = value.get("accounts") if isinstance(value, dict) else value
    if not isinstance(accounts, list) or not all(isinstance(item, dict) for item in accounts):
        raise ExtractorEvaluationError("local account catalog is invalid")
    entries = []
    for item in accounts:
        try:
            entries.append(
                AccountCatalogEntry(
                    account_id=item["account_id"],
                    account_type=item["account_type"],
                    aliases=tuple(item["aliases"]),
                    owned=item.get("owned", True),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtractorEvaluationError("local account catalog is invalid") from exc
    return entries, file_sha256(resolved)


def _resolve_for_evaluation(
    reference: Any, catalog: Sequence[AccountCatalogEntry] | None
) -> dict[str, Any] | None:
    if catalog is None:
        return None
    resolution = resolve_account(reference, catalog)
    return {"status": resolution.status.value}


def _account_compare(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_account_reference(str(value))
    return normalized or None


def _counterparty_compare(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_counterparty(str(value))
    return normalized or None


def _build_report(
    *,
    suite: str,
    provenance: Mapping[str, Any],
    configuration_hash: str,
    capabilities: Mapping[str, Any],
    rows: Sequence[EvaluationRow],
    records: Mapping[str, Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    ordered = [records[row.row_id] for row in rows if row.row_id in records]
    adjusted_capabilities = dict(capabilities)
    if provenance["assets"].get("account_catalog_sha256"):
        adjusted_capabilities["account_resolution"] = True
    return {
        "contract": EVALUATION_CONTRACT,
        "status": status,
        "suite": suite,
        "configuration_sha256": configuration_hash,
        "provenance": dict(provenance),
        "capabilities": adjusted_capabilities,
        "production_readiness_claim": False,
        "counts": {"completed": len(ordered), "total": len(rows)},
        "metrics": _metrics(ordered),
        "groups": _group_metrics(ordered, suite=suite),
    }


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [record for record in records if record.get("gold") is not None]
    gold_posted = sum(record["gold"]["decision"] == "posted" for record in labelled)
    predicted_posted = sum(record["outcome"] == "posted" for record in labelled)
    true_posted = sum(
        record["gold"]["decision"] == "posted" and record["outcome"] == "posted"
        for record in labelled
    )
    precision = _ratio(true_posted, predicted_posted)
    recall = _ratio(true_posted, gold_posted)
    f1 = None
    if precision is not None and recall is not None:
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    strict = [
        _strict_match(record)
        for record in labelled
        if _strict_gold_supported(record["gold"])
    ]
    grounding = [
        _grounding_valid(record)
        for record in records
        if record.get("outcome") == "posted"
    ]
    resolutions = [
        record["account_resolution"]["status"]
        for record in records
        if record.get("account_resolution") is not None
    ]
    outcome_names = ("none", "abstain", "malformed", "runtime_failure")
    outcomes = {
        name: sum(record["outcome"] == name for record in records)
        for name in outcome_names
    }
    reason_code_counts: dict[str, int] = {}
    for record in records:
        reason = record.get("reason_code")
        if reason:
            reason_code_counts[str(reason)] = reason_code_counts.get(str(reason), 0) + 1
    count = len(records)
    return {
        "labelled_count": len(labelled),
        "transaction": {"precision": precision, "recall": recall, "f1": f1},
        "strict_posted_exact_success": {
            "count": sum(strict),
            "eligible_count": len(strict),
            "rate": _ratio(sum(strict), len(strict)),
        },
        "amount_accuracy": _field_accuracy(labelled, "amount"),
        "direction_accuracy": _field_accuracy(labelled, "direction"),
        "account_reference_accuracy": _field_accuracy(labelled, "account"),
        "counterparty_accuracy": _field_accuracy(labelled, "counterparty"),
        "amount_grounding_validity": {
            "count": sum(grounding),
            "eligible_count": len(grounding),
            "rate": _ratio(sum(grounding), len(grounding)),
        },
        "account_resolution": {
            "eligible_count": len(resolutions),
            "success_rate": _ratio(
                resolutions.count("uniquely_resolved"), len(resolutions)
            ),
            "ambiguity_rate": _ratio(resolutions.count("ambiguous"), len(resolutions)),
        },
        "outcome_rates": {
            **{name: _ratio(value, count) for name, value in outcomes.items()},
            "review": _ratio(
                outcomes["abstain"]
                + outcomes["malformed"]
                + outcomes["runtime_failure"],
                count,
            ),
        },
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "latency_ms": _latency_summary(
            [float(record["latency_ms"]) for record in records]
        ),
    }


def _strict_gold_supported(gold: Mapping[str, Any]) -> bool:
    return gold.get("decision") == "posted" and all(
        gold.get(name) is not None
        for name in (
            "minor_units",
            "currency",
            "direction",
            "account_reference",
        )
    )


def _strict_match(record: Mapping[str, Any]) -> bool:
    gold = record["gold"]
    prediction = record.get("prediction")
    if record.get("outcome") != "posted" or not isinstance(prediction, Mapping):
        return False
    fields_match = (
        prediction.get("minor_units") == gold.get("minor_units")
        and prediction.get("currency") == gold.get("currency")
        and prediction.get("direction") == gold.get("direction")
        and prediction.get("account_reference") == gold.get("account_reference")
        and prediction.get("counterparty") == gold.get("counterparty")
    )
    gold_spans = gold.get("spans")
    if not fields_match or gold_spans is None:
        return fields_match
    prediction_spans = prediction.get("spans")
    if not isinstance(prediction_spans, Mapping):
        return False
    return all(
        expected is None or prediction_spans.get(name) == expected
        for name, expected in gold_spans.items()
    )


def _field_accuracy(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    eligible = []
    for record in records:
        gold = record["gold"]
        if gold["decision"] != "posted":
            continue
        if field == "amount" and gold.get("minor_units") is None:
            continue
        if field == "direction" and gold.get("direction") is None:
            continue
        if field == "account" and gold.get("account_reference") is None:
            continue
        if field == "counterparty" and not gold.get("counterparty_available"):
            continue
        eligible.append(record)
    correct = 0
    for record in eligible:
        prediction = record.get("prediction") or {}
        gold = record["gold"]
        if field == "amount":
            matched = (
                prediction.get("minor_units") == gold.get("minor_units")
                and prediction.get("currency") == gold.get("currency")
            )
        elif field == "direction":
            matched = prediction.get("direction") == gold.get("direction")
        elif field == "account":
            matched = prediction.get("account_reference") == gold.get(
                "account_reference"
            )
        else:
            matched = prediction.get("counterparty") == gold.get("counterparty")
        correct += int(matched)
    return {
        "count": correct,
        "eligible_count": len(eligible),
        "rate": _ratio(correct, len(eligible)),
    }


def _grounding_valid(record: Mapping[str, Any]) -> bool:
    prediction = record.get("prediction") or {}
    span = (prediction.get("spans") or {}).get("amount")
    return isinstance(span, Mapping) and isinstance(span.get("text"), str)


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _group_metrics(
    records: Sequence[Mapping[str, Any]], *, suite: str
) -> dict[str, Any]:
    if suite == "grandfathered":
        return {
            "available": False,
            "reason": "grandfathered fixture has no approved evaluation grouping labels",
        }
    result: dict[str, Any] = {
        "available": True,
        "small_group_minimum": _SMALL_GROUP_MINIMUM,
    }
    for dimension in ("sender_family", "template_family"):
        buckets: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            group = record.get("groups", {}).get(dimension)
            if group:
                buckets.setdefault(str(group), []).append(record)
        visible = {}
        suppressed_groups = 0
        suppressed_rows = 0
        for group, values in sorted(buckets.items()):
            if suite == "private-canonical" and len(values) < _SMALL_GROUP_MINIMUM:
                suppressed_groups += 1
                suppressed_rows += len(values)
                continue
            aggregate = _metrics(values)
            visible[group] = {
                "count": len(values),
                "transaction": aggregate["transaction"],
                "strict_posted_exact_success": aggregate[
                    "strict_posted_exact_success"
                ],
                "outcome_rates": aggregate["outcome_rates"],
            }
        result[dimension] = {
            "groups": visible,
            "suppressed_group_count": suppressed_groups,
            "suppressed_row_count": suppressed_rows,
        }
    return result


def _aggregate_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Strip report content down to the aggregate-only CLI response."""

    return {
        "status": report["status"],
        "suite": report["suite"],
        "completed_rows": report["counts"]["completed"],
        "total_rows": report["counts"]["total"],
        "configuration_sha256": report["configuration_sha256"],
        "metrics": report["metrics"],
        "production_readiness_claim": False,
    }
