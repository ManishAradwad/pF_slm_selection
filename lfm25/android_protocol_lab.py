"""Fail-closed Android Phase D prompt/output-protocol laboratory.

The laboratory is deliberately separate from the Android application.  It maps
family-specific model outputs into the frozen Semantic V2 reference, scores only
committed invented fixtures, and emits aggregate-only evidence.  Importing this
module never loads a model or a heavyweight runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from .evaluation_v2 import (
    EVALUATION_CONTRACT_ID,
    EVALUATION_CONTRACT_VERSION,
    assess_candidate,
    score_evaluation_package,
    split_manifest_sha256,
)
from .semantic_v2 import (
    SEMANTIC_CONTRACT_ID,
    SEMANTIC_CONTRACT_VERSION,
    SemanticV2Error,
    derive_currency_from_amount_evidence,
    derive_decimal_text_from_amount_evidence,
    derive_direction_from_evidence,
    derive_minor_units,
    inject_source_timestamp,
    project_initial_auto_post,
    slice_utf8_evidence,
    validate_semantic_v2,
    EvidenceSpan,
)
from .workbench_v2 import (
    resolved_annotation_rows,
    validate_annotation_package,
    workbench_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAB_CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "configs/contracts/pocketfinancer-android-phase-d-lab-v1.json"
)
LAB_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "configs/experiments/pocketfinancer-android-phase-d-v1.json"
)
RUNTIME_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "configs/experiments/pocketfinancer-android-phase-d-runtime-v1.json"
)
SYNTHETIC_FIXTURE_PATH = (
    REPOSITORY_ROOT
    / "tests/fixtures/pocketfinancer_workbench_v2_synthetic.json"
)
SEMANTIC_FIXTURE_PATH = (
    REPOSITORY_ROOT
    / "tests/fixtures/pocketfinancer_semantic_v2_synthetic_golden.json"
)

LAB_ID = "pocketfinancer-android-phase-d-protocol-lab"
LAB_VERSION = 1
BASELINE_ID = "pocketfinancer-android-552ffbdf-phase-c"
BASELINE_SHA256 = "9274e5a63524b46bb4149e11d5190bae4ebcfef15170a69af77ad050b31167fc"
ANDROID_COMMIT = "552ffbdfbd41773980aa249789b0cb508fdb19fd"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,127}\Z")
_QUANTIZATIONS = {"Q4_K_M", "Q8_0"}
_TIMESTAMP_KEYS = {
    "date",
    "datetime",
    "timestamp",
    "transaction_date",
    "received_at_epoch_ms",
    "received_at_provenance",
}
_FORBIDDEN_RESULT_KEYS = {
    "message",
    "sender",
    "raw_output",
    "model_output",
    "prompt_text",
    "row_id",
    "device_serial",
    "build_fingerprint",
}


class AndroidProtocolLabError(ValueError):
    """A Phase D manifest, protocol output, or result violates the lab contract."""


class _ProtocolShapeError(ValueError):
    pass


@dataclass(frozen=True)
class ProtocolParseResult:
    parse_status: str
    semantic_record: Mapping[str, Any] | None
    failure_code: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AndroidProtocolLabError(f"invalid JSON resource: {path}") from error


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AndroidProtocolLabError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise AndroidProtocolLabError(f"{path} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise AndroidProtocolLabError(f"{path} has invalid keys")


def _text(value: Any, path: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise AndroidProtocolLabError(f"{path} must be non-empty text")
    if pattern is not None and not pattern.fullmatch(value):
        raise AndroidProtocolLabError(f"{path} has an invalid format")
    return value


def _sha256(value: Any, path: str) -> str:
    return _text(value, path, pattern=_SHA256_RE)


def _relative_path(value: Any, path: str) -> Path:
    text = _text(value, path)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AndroidProtocolLabError(f"{path} must stay inside the repository")
    return candidate


def lab_contract() -> dict[str, Any]:
    value = _object(_read_json(LAB_CONTRACT_PATH), "lab_contract")
    if (
        value.get("contract_id") != "pocketfinancer_android_phase_d_protocol_lab"
        or value.get("contract_version") != 1
        or value.get("status") != "phase_d_evaluation_only_frozen_contract"
    ):
        raise AndroidProtocolLabError("Phase D lab contract identity is invalid")
    if dict(_object(value.get("required_baseline_binding"), "required_baseline_binding")) != {
        "baseline_id": BASELINE_ID,
        "baseline_manifest_sha256": BASELINE_SHA256,
        "android_commit": ANDROID_COMMIT,
    }:
        raise AndroidProtocolLabError("Phase D baseline binding has drifted")
    return dict(value)


def _validate_resource(value: Any, path: str) -> dict[str, str]:
    resource = _object(value, path)
    _exact_keys(resource, {"path", "sha256"}, path)
    relative = _relative_path(resource["path"], f"{path}.path")
    expected = _sha256(resource["sha256"], f"{path}.sha256")
    absolute = REPOSITORY_ROOT / relative
    if not absolute.is_file() or sha256_file(absolute) != expected:
        raise AndroidProtocolLabError(f"{path} hash verification failed")
    return {"path": relative.as_posix(), "sha256": expected}


def _validate_model(value: Any, path: str) -> dict[str, Any]:
    model = _object(value, path)
    _exact_keys(
        model,
        {
            "artifact_name",
            "artifact_path",
            "artifact_sha256",
            "artifact_identity_basis",
            "upstream_repository",
            "model_revision",
            "quantization",
            "thinking_enabled",
            "gguf_architecture",
            "gguf_name",
            "chat_template_sha256",
        },
        path,
    )
    artifact_path = _relative_path(model["artifact_path"], f"{path}.artifact_path")
    artifact_name = _text(model["artifact_name"], f"{path}.artifact_name")
    if artifact_path.name != artifact_name or artifact_path.parts[:1] != ("MODELS",):
        raise AndroidProtocolLabError(f"{path} has an invalid local artifact path")
    artifact_sha256 = _sha256(model["artifact_sha256"], f"{path}.artifact_sha256")
    if model["artifact_identity_basis"] != "local_sha256_verified_before_inference":
        raise AndroidProtocolLabError(f"{path} has an unsupported artifact identity basis")
    model_revision = _text(model["model_revision"], f"{path}.model_revision")
    if model_revision != f"gguf-sha256:{artifact_sha256}":
        raise AndroidProtocolLabError(f"{path}.model_revision must bind the GGUF hash")
    quantization = _text(model["quantization"], f"{path}.quantization")
    if quantization not in _QUANTIZATIONS:
        raise AndroidProtocolLabError(f"{path}.quantization is unsupported")
    if not isinstance(model["thinking_enabled"], bool):
        raise AndroidProtocolLabError(f"{path}.thinking_enabled must be boolean")
    return {
        **dict(model),
        "artifact_path": artifact_path.as_posix(),
        "artifact_sha256": artifact_sha256,
        "chat_template_sha256": _sha256(
            model["chat_template_sha256"], f"{path}.chat_template_sha256"
        ),
    }


def _validate_profile(value: Any, path: str) -> dict[str, Any]:
    profile = _object(value, path)
    _exact_keys(
        profile,
        {
            "experiment_profile_id",
            "runtime_variant_id",
            "app_tier_id",
            "model_family",
            "model",
            "protocol_id",
            "prompt_profile",
            "selected_profile_id",
        },
        path,
    )
    profile_id = _text(
        profile["experiment_profile_id"],
        f"{path}.experiment_profile_id",
        pattern=_ID_RE,
    )
    runtime_variant = _text(profile["runtime_variant_id"], f"{path}.runtime_variant_id")
    contract = lab_contract()
    if runtime_variant not in contract["runtime_variant_ids"]:
        raise AndroidProtocolLabError(f"{path}.runtime_variant_id is unsupported")
    family = _text(profile["model_family"], f"{path}.model_family")
    expected_family = "qwen3" if runtime_variant.startswith("qwen3-") else "gemma4"
    if family != expected_family:
        raise AndroidProtocolLabError(f"{path}.model_family does not match the variant")
    protocol_id = _text(profile["protocol_id"], f"{path}.protocol_id")
    if protocol_id not in contract["protocol_ids"]:
        raise AndroidProtocolLabError(f"{path}.protocol_id is unsupported")
    prompt = _validate_resource(profile["prompt_profile"], f"{path}.prompt_profile")
    prompt_value = _object(
        _read_json(REPOSITORY_ROOT / prompt["path"]), f"{path}.prompt_profile.value"
    )
    if (
        prompt_value.get("model_family") != family
        or prompt_value.get("protocol_id") != protocol_id
    ):
        raise AndroidProtocolLabError(f"{path}.prompt_profile does not match the profile")
    if profile["selected_profile_id"] is not None:
        raise AndroidProtocolLabError(f"{path}.selected_profile_id must remain null")
    return {
        "experiment_profile_id": profile_id,
        "runtime_variant_id": runtime_variant,
        "app_tier_id": _text(profile["app_tier_id"], f"{path}.app_tier_id"),
        "model_family": family,
        "model": _validate_model(profile["model"], f"{path}.model"),
        "protocol_id": protocol_id,
        "prompt_profile": prompt,
        "selected_profile_id": None,
    }


def validate_lab_manifest(value: Any) -> dict[str, Any]:
    manifest = _object(value, "lab_manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "lab_id",
            "lab_version",
            "status",
            "bindings",
            "resources",
            "comparison",
            "profiles",
            "selection",
        },
        "lab_manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["lab_id"] != LAB_ID
        or manifest["lab_version"] != LAB_VERSION
        or manifest["status"] != "phase_d_evaluation_only"
    ):
        raise AndroidProtocolLabError("Phase D lab manifest identity is invalid")

    bindings = _object(manifest["bindings"], "bindings")
    _exact_keys(
        bindings,
        {
            "baseline_id",
            "baseline_manifest_sha256",
            "android_commit",
            "locked_production_profile_revision",
        },
        "bindings",
    )
    if dict(bindings) != {
        "baseline_id": BASELINE_ID,
        "baseline_manifest_sha256": BASELINE_SHA256,
        "android_commit": ANDROID_COMMIT,
        "locked_production_profile_revision": "a9b7df44be2183daac3a05cadbfd40b8f309cd4b",
    }:
        raise AndroidProtocolLabError("Phase D manifest baseline binding has drifted")

    resources = _object(manifest["resources"], "resources")
    expected_resources = {
        "lab_contract",
        "runtime_profile",
        "parser",
        "runner",
        "evaluator",
        "synthetic_fixture",
        "semantic_conformance_fixture",
        "decision_policy",
    }
    _exact_keys(resources, expected_resources, "resources")
    normalized_resources = {
        name: _validate_resource(resource, f"resources.{name}")
        for name, resource in resources.items()
    }

    comparison = _object(manifest["comparison"], "comparison")
    _exact_keys(
        comparison,
        {
            "material_variable",
            "held_constant",
            "cross_runtime_pooling",
            "universal_protocol_claim",
        },
        "comparison",
    )
    if (
        comparison["material_variable"] != "prompt_and_output_protocol"
        or comparison["cross_runtime_pooling"] != "forbidden"
        or comparison["universal_protocol_claim"] != "forbidden"
        or not isinstance(comparison["held_constant"], list)
    ):
        raise AndroidProtocolLabError("Phase D comparison policy is invalid")

    profiles = [
        _validate_profile(item, f"profiles[{index}]")
        for index, item in enumerate(_array(manifest["profiles"], "profiles"))
    ]
    contract = lab_contract()
    if len(profiles) != contract["comparison_policy"]["profiles_required"]:
        raise AndroidProtocolLabError("Phase D must declare exactly ten profiles")
    profile_ids = {item["experiment_profile_id"] for item in profiles}
    if len(profile_ids) != len(profiles):
        raise AndroidProtocolLabError("Phase D experiment profile IDs must be unique")
    pairs = {(item["runtime_variant_id"], item["protocol_id"]) for item in profiles}
    expected_pairs = {
        (variant, protocol)
        for variant in contract["runtime_variant_ids"]
        for protocol in contract["protocol_ids"]
    }
    if pairs != expected_pairs:
        raise AndroidProtocolLabError("each runtime variant requires Direct and Candidate profiles")

    selection = _object(manifest["selection"], "selection")
    _exact_keys(
        selection,
        {
            "selected_profile_id",
            "direct_v2_status",
            "candidate_v2_status",
            "production_defaults_changed",
            "phase_e_started",
        },
        "selection",
    )
    if dict(selection) != {
        "selected_profile_id": None,
        "direct_v2_status": "unselected_hypothesis",
        "candidate_v2_status": "unselected_hypothesis",
        "production_defaults_changed": False,
        "phase_e_started": False,
    }:
        raise AndroidProtocolLabError("Phase D selection state must remain unselected")
    return {
        **dict(manifest),
        "bindings": dict(bindings),
        "resources": normalized_resources,
        "comparison": dict(comparison),
        "profiles": profiles,
        "selection": dict(selection),
        "manifest_sha256": sha256_json(manifest),
    }


def load_lab_manifest() -> dict[str, Any]:
    return validate_lab_manifest(_read_json(LAB_MANIFEST_PATH))


def load_prompt_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    path = REPOSITORY_ROOT / profile["prompt_profile"]["path"]
    value = _object(_read_json(path), "prompt_profile")
    _exact_keys(
        value,
        {
            "schema_version",
            "prompt_profile_id",
            "model_family",
            "protocol_id",
            "chat_messages",
            "output_contract",
        },
        "prompt_profile",
    )
    if (
        value["schema_version"] != 1
        or value["model_family"] != profile["model_family"]
        or value["protocol_id"] != profile["protocol_id"]
    ):
        raise AndroidProtocolLabError("prompt profile identity does not match experiment profile")
    messages = _object(value["chat_messages"], "prompt_profile.chat_messages")
    _exact_keys(messages, {"system", "user_template"}, "prompt_profile.chat_messages")
    if messages["system"] is not None and not isinstance(messages["system"], str):
        raise AndroidProtocolLabError("prompt system message must be text or null")
    template = _text(messages["user_template"], "prompt_profile.user_template")
    if template.count("{{MESSAGE}}") != 1:
        raise AndroidProtocolLabError("prompt template requires exactly one message placeholder")
    return dict(value)


def render_chat_messages(prompt_profile: Mapping[str, Any], message: str) -> list[dict[str, str]]:
    if not isinstance(message, str):
        raise AndroidProtocolLabError("message must be text")
    chat = _object(prompt_profile["chat_messages"], "prompt_profile.chat_messages")
    user = _text(chat["user_template"], "prompt_profile.user_template").replace(
        "{{MESSAGE}}", message
    )
    messages: list[dict[str, str]] = []
    if chat["system"] is not None:
        messages.append({"role": "system", "content": chat["system"]})
    messages.append({"role": "user", "content": user})
    return messages


def _contains_timestamp_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _TIMESTAMP_KEYS:
                return True
            if _contains_timestamp_field(item):
                return True
    elif isinstance(value, list):
        return any(_contains_timestamp_field(item) for item in value)
    return False


def _clean_output(raw_output: Any) -> tuple[str | None, str | None]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None, "empty_output"
    cleaned = raw_output.strip()
    while cleaned.startswith("<think>"):
        end = cleaned.find("</think>")
        if end < 0:
            return None, "unclosed_thinking_block"
        cleaned = cleaned[end + len("</think>") :].strip()
    if "<think>" in cleaned or "</think>" in cleaned:
        return None, "non_json_output"
    return cleaned, None


def _json_output(raw_output: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    cleaned, failure = _clean_output(raw_output)
    if failure is not None:
        return None, failure
    assert cleaned is not None
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, "non_json_output"
    if not isinstance(value, Mapping):
        return None, "json_root_not_object"
    if "source_metadata" in value:
        return None, "forbidden_source_metadata"
    if _contains_timestamp_field(value):
        return None, "forbidden_timestamp_field"
    return value, None


def _candidate_exact_keys(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise _ProtocolShapeError(f"{path} has invalid keys")
    return value


def _candidate_text(value: Any, allowed: Sequence[str], path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _ProtocolShapeError(f"{path} has an unsupported value")
    return value


def _candidate_span(value: Any, message: str, path: str) -> tuple[dict[str, int], str]:
    span_value = _candidate_exact_keys(
        value, {"start_utf8_byte", "end_utf8_byte"}, path
    )
    start = span_value["start_utf8_byte"]
    end = span_value["end_utf8_byte"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise _ProtocolShapeError(f"{path} has invalid byte offsets")
    span = EvidenceSpan(start, end)
    text = slice_utf8_evidence(message, span)
    return {"start_utf8_byte": start, "end_utf8_byte": end}, text


def _candidate_core(value: Mapping[str, Any], message: str) -> dict[str, Any]:
    root = _candidate_exact_keys(
        value, {"scope", "posting_status", "event_cardinality", "events"}, "candidate"
    )
    scope = _candidate_text(root["scope"], ("bank_card", "wallet_bnpl", "other"), "scope")
    posting_status = _candidate_text(
        root["posting_status"], ("posted", "not_posted"), "posting_status"
    )
    cardinality = _candidate_text(
        root["event_cardinality"], ("none", "single", "multiple"), "event_cardinality"
    )
    if not isinstance(root["events"], list):
        raise _ProtocolShapeError("events must be an array")
    events = []
    for index, item in enumerate(root["events"]):
        event = _candidate_exact_keys(
            item,
            {"amount_evidence", "direction_evidence", "account_evidence", "counterparty"},
            f"events[{index}]",
        )
        amount = None
        if event["amount_evidence"] is not None:
            evidence, evidence_text = _candidate_span(
                event["amount_evidence"], message, f"events[{index}].amount_evidence"
            )
            decimal_text = derive_decimal_text_from_amount_evidence(evidence_text)
            currency = derive_currency_from_amount_evidence(evidence_text)
            amount = {
                "decimal_text": decimal_text,
                "currency": currency,
                "minor_units": derive_minor_units(decimal_text, currency),
                "evidence": evidence,
            }

        direction = None
        if event["direction_evidence"] is not None:
            evidence, evidence_text = _candidate_span(
                event["direction_evidence"], message, f"events[{index}].direction_evidence"
            )
            direction = {
                "value": derive_direction_from_evidence(evidence_text).value,
                "evidence": evidence,
            }

        account = None
        if event["account_evidence"] is not None:
            evidence, evidence_text = _candidate_span(
                event["account_evidence"], message, f"events[{index}].account_evidence"
            )
            account = {"value": evidence_text, "evidence": evidence}

        counterparty_value = event["counterparty"]
        if not isinstance(counterparty_value, Mapping):
            raise _ProtocolShapeError(f"events[{index}].counterparty must be an object")
        state = counterparty_value.get("state")
        if state == "absent":
            _candidate_exact_keys(
                counterparty_value, {"state"}, f"events[{index}].counterparty"
            )
            counterparty = {"state": "absent"}
        elif state == "present":
            present = _candidate_exact_keys(
                counterparty_value,
                {"state", "evidence"},
                f"events[{index}].counterparty",
            )
            evidence, evidence_text = _candidate_span(
                present["evidence"], message, f"events[{index}].counterparty.evidence"
            )
            counterparty = {
                "state": "present",
                "value": evidence_text,
                "evidence": evidence,
            }
        else:
            raise _ProtocolShapeError(f"events[{index}].counterparty.state is unsupported")

        events.append(
            {
                "event_id": f"event_{index + 1}",
                "amount": amount,
                "direction": direction,
                "account": account,
                "counterparty": counterparty,
            }
        )
    return {
        "semantic_contract_id": SEMANTIC_CONTRACT_ID,
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "scope": scope,
        "posting_status": posting_status,
        "event_cardinality": cardinality,
        "events": events,
    }


def _semantic_failure_code(error: SemanticV2Error) -> str:
    message = str(error).casefold()
    if "evidence" in message or "utf-8" in message:
        return "ungrounded_evidence"
    return "semantic_validation_failed"


def parse_protocol_output(
    raw_output: Any,
    *,
    protocol_id: str,
    message: str,
    source_metadata: Mapping[str, Any],
) -> ProtocolParseResult:
    """Map one family-specific output into Semantic V2 or fail closed."""

    value, failure = _json_output(raw_output)
    if failure is not None:
        return ProtocolParseResult("invalid", None, failure)
    assert value is not None
    try:
        if protocol_id == "direct_v2":
            core = dict(value)
        elif protocol_id == "candidate_v2":
            core = _candidate_core(value, message)
        else:
            raise AndroidProtocolLabError("unsupported protocol_id")
        complete = inject_source_timestamp(
            core,
            received_at_epoch_ms=source_metadata["received_at_epoch_ms"],
            received_at_provenance=source_metadata["received_at_provenance"],
        )
        validate_semantic_v2(complete, message=message)
    except _ProtocolShapeError:
        return ProtocolParseResult("invalid", None, "protocol_shape_invalid")
    except (KeyError, TypeError):
        return ProtocolParseResult("invalid", None, "protocol_shape_invalid")
    except SemanticV2Error as error:
        return ProtocolParseResult("invalid", None, _semantic_failure_code(error))
    return ProtocolParseResult("valid", complete, None)


def candidate_output_from_semantic_core(core: Mapping[str, Any]) -> dict[str, Any]:
    """Create a Candidate V2 protocol vector from an invented Semantic V2 core."""

    events = []
    for event in core["events"]:
        amount = event["amount"]
        direction = event["direction"]
        account = event["account"]
        counterparty = event["counterparty"]
        if counterparty["state"] == "present":
            candidate_counterparty = {
                "state": "present",
                "evidence": dict(counterparty["evidence"]),
            }
        else:
            candidate_counterparty = {"state": "absent"}
        events.append(
            {
                "amount_evidence": None if amount is None else dict(amount["evidence"]),
                "direction_evidence": (
                    None if direction is None else dict(direction["evidence"])
                ),
                "account_evidence": None if account is None else dict(account["evidence"]),
                "counterparty": candidate_counterparty,
            }
        )
    return {
        "scope": core["scope"],
        "posting_status": core["posting_status"],
        "event_cardinality": core["event_cardinality"],
        "events": events,
    }


def run_parser_conformance() -> dict[str, Any]:
    """Exercise both adapters against every frozen invented Semantic V2 vector."""

    fixture = _object(_read_json(SEMANTIC_FIXTURE_PATH), "semantic_fixture")
    vectors = _array(fixture.get("vectors"), "semantic_fixture.vectors")
    results: dict[str, dict[str, int]] = {}
    for protocol_id in ("direct_v2", "candidate_v2"):
        passed = 0
        uncaught = 0
        for vector in vectors:
            try:
                core = vector["semantic_core"]
                raw = (
                    core
                    if protocol_id == "direct_v2"
                    else candidate_output_from_semantic_core(core)
                )
                parsed = parse_protocol_output(
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                    protocol_id=protocol_id,
                    message=vector["message"],
                    source_metadata=vector["source_metadata"],
                )
                expected_mapping = inject_source_timestamp(
                    core,
                    received_at_epoch_ms=vector["source_metadata"]["received_at_epoch_ms"],
                    received_at_provenance=vector["source_metadata"][
                        "received_at_provenance"
                    ],
                )
                expected = validate_semantic_v2(expected_mapping, message=vector["message"])
                actual = (
                    None
                    if parsed.semantic_record is None
                    else validate_semantic_v2(
                        parsed.semantic_record, message=vector["message"]
                    )
                )
                passed += int(parsed.parse_status == "valid" and actual == expected)
            except Exception:
                uncaught += 1
        results[protocol_id] = {
            "passed": passed,
            "total": len(vectors),
            "uncaught_exceptions": uncaught,
        }
    return {
        "fixture_sha256": sha256_file(SEMANTIC_FIXTURE_PATH),
        "protocols": results,
    }


def verify_local_artifact(profile: Mapping[str, Any]) -> dict[str, Any]:
    model = _object(profile["model"], "profile.model")
    path = REPOSITORY_ROOT / model["artifact_path"]
    if not path.is_file():
        return {"status": "not_measured", "gap_code": "gguf_artifact_missing"}
    actual = sha256_file(path)
    if actual != model["artifact_sha256"]:
        return {"status": "not_measured", "gap_code": "gguf_artifact_hash_mismatch"}
    return {"status": "verified", "sha256": actual, "path": path}


def _runtime_environment_sha256(runtime: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "runtime_profile_sha256": sha256_file(RUNTIME_PROFILE_PATH),
            "host_gguf_runtime": runtime,
        }
    )


def _prediction_package(
    *,
    profile: Mapping[str, Any],
    prompt_profile: Mapping[str, Any],
    annotation_package: Mapping[str, Any],
    rows: list[dict[str, Any]],
    conformance: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    annotation_summary = validate_annotation_package(annotation_package)
    semantic = workbench_contract()["semantic_contract"]
    baseline = _object(
        _read_json(
            REPOSITORY_ROOT
            / "configs/baselines/pocketfinancer-android-552ffbdf-phase-c.json"
        ),
        "android_baseline",
    )
    filter_sha256 = baseline["source_snapshot"]["files_sha256"][
        "pipeline/src/main/java/com/pocketfinancer/pipeline/SmsFilterPipeline.kt"
    ]
    return {
        "evaluation_contract": {
            "id": EVALUATION_CONTRACT_ID,
            "version": EVALUATION_CONTRACT_VERSION,
        },
        "semantic_contract": {
            "id": semantic["id"],
            "version": semantic["version"],
            "schema_sha256": semantic["schema_sha256"],
            "reference_sha256": semantic["reference_sha256"],
            "conformance_sha256": semantic["conformance_sha256"],
        },
        "profile": {
            "profile_id": profile["experiment_profile_id"],
            "platform": "host_gguf",
            "runtime_variant": profile["runtime_variant_id"],
            "protocol_id": profile["protocol_id"],
        },
        "provenance": {
            "annotation_package_sha256": annotation_summary["package_sha256"],
            "split_manifest_sha256": split_manifest_sha256(annotation_package),
            "semantic_schema_sha256": semantic["schema_sha256"],
            "semantic_reference_sha256": semantic["reference_sha256"],
            "semantic_conformance_sha256": semantic["conformance_sha256"],
            "prompt_sha256": profile["prompt_profile"]["sha256"],
            "decode_settings_sha256": sha256_file(RUNTIME_PROFILE_PATH),
            "parser_sha256": sha256_file(Path(__file__)),
            "filter_sha256": filter_sha256,
            "runtime_environment_sha256": _runtime_environment_sha256(runtime),
            "evaluator_id": EVALUATION_CONTRACT_ID,
            "model_runtime_id": "llama-cpp-python-0.3.20-cpu",
            "model_revision": profile["model"]["model_revision"],
            "prompt_profile_id": prompt_profile["prompt_profile_id"],
            "chat_template_sha256": profile["model"]["chat_template_sha256"],
            "quantization": profile["model"]["quantization"],
            "selected_checkpoint": None,
            "evaluator_version": EVALUATION_CONTRACT_VERSION,
            "seed": runtime["seed"],
        },
        "conformance": dict(conformance),
        "privacy_provenance": {
            "local_only": True,
            "protected_split_locked": True,
            "contract_hashes_verified": True,
            "annotation_and_split_hashes_verified": True,
            "profile_provenance_complete": True,
        },
        "operational_budget": {
            "budget_id": None,
            "budget_sha256": None,
            "frozen_before_measurement": False,
            "evidence_class": "host_gguf",
            "baseline_profile_id": None,
            "hard_limits_passed": False,
        },
        "rows": rows,
    }


def run_host_gguf_profile(
    llama: Any,
    *,
    profile: Mapping[str, Any],
    manifest: Mapping[str, Any],
    annotation_package: Mapping[str, Any],
    runtime_profile: Mapping[str, Any],
    conformance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one non-pooled profile without retaining prompts or model output."""

    prompt_profile = load_prompt_profile(profile)
    runtime = _object(runtime_profile["host_gguf_runtime"], "host_gguf_runtime")
    rows = []
    failure_taxonomy = {code: 0 for code in lab_contract()["parser_failure_codes"]}
    max_tokens = (
        runtime["qwen_max_completion_tokens"]
        if profile["model_family"] == "qwen3"
        else runtime["gemma_max_completion_tokens"]
    )
    for resolved in resolved_annotation_rows(annotation_package, split="protected_test"):
        started = time.perf_counter()
        available = True
        try:
            response = llama.create_chat_completion(
                messages=render_chat_messages(prompt_profile, resolved.message),
                temperature=runtime["temperature"],
                top_p=runtime["top_p"],
                top_k=runtime["top_k"],
                min_p=runtime["min_p"],
                typical_p=runtime["typical_p"],
                repeat_penalty=runtime["repeat_penalty"],
                seed=runtime["seed"],
                max_tokens=max_tokens,
            )
            raw_output = response["choices"][0]["message"].get("content") or ""
            parsed = parse_protocol_output(
                raw_output,
                protocol_id=profile["protocol_id"],
                message=resolved.message,
                source_metadata=resolved.gold_mapping["source_metadata"],
            )
        except Exception:
            available = False
            parsed = ProtocolParseResult("invalid", None, "runtime_error")
        latency_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        if parsed.failure_code is not None:
            failure_taxonomy[parsed.failure_code] += 1
        decision = "reject"
        if parsed.semantic_record is not None:
            record = validate_semantic_v2(
                parsed.semantic_record, message=resolved.message
            )
            decision = "auto_post" if project_initial_auto_post(record).eligible else "review"
        rows.append(
            {
                "row_id": resolved.row_id,
                "parse_status": parsed.parse_status,
                "semantic_record": (
                    None if parsed.semantic_record is None else dict(parsed.semantic_record)
                ),
                "decision": decision,
                "failure_code": parsed.failure_code,
                "measurements": {
                    "latency_ms": latency_ms,
                    "peak_memory_bytes": None,
                    "battery_impact_millipercent": None,
                    "available": available,
                    "recovery_attempted": None,
                    "recovery_succeeded": None,
                },
            }
        )

    prediction = _prediction_package(
        profile=profile,
        prompt_profile=prompt_profile,
        annotation_package=annotation_package,
        rows=rows,
        conformance=conformance,
        runtime=runtime,
    )
    report = score_evaluation_package(annotation_package, prediction)
    assessment = assess_candidate(report, baseline_report=None)
    return {
        "experiment_profile_id": profile["experiment_profile_id"],
        "runtime_variant_id": profile["runtime_variant_id"],
        "model_family": profile["model_family"],
        "protocol_id": profile["protocol_id"],
        "selected_profile_id": None,
        "provenance": {
            "android_commit": ANDROID_COMMIT,
            "baseline_id": BASELINE_ID,
            "baseline_manifest_sha256": BASELINE_SHA256,
            "model_revision": profile["model"]["model_revision"],
            "artifact_sha256": profile["model"]["artifact_sha256"],
            "chat_template_sha256": profile["model"]["chat_template_sha256"],
            "prompt_profile_id": prompt_profile["prompt_profile_id"],
            "prompt_sha256": profile["prompt_profile"]["sha256"],
            "parser_sha256": sha256_file(Path(__file__)),
            "runtime_profile_sha256": sha256_file(RUNTIME_PROFILE_PATH),
            "evaluator_sha256": manifest["resources"]["evaluator"]["sha256"],
            "fixture_sha256": manifest["resources"]["synthetic_fixture"]["sha256"],
        },
        "evidence": {
            "source_static": {
                "status": "measured",
                "conformance": dict(conformance),
            },
            "host_hf": {
                "status": "not_applicable",
                "gap_codes": ["hf_artifact_not_in_phase_d_scope"],
            },
            "host_gguf": {
                "status": "measured",
                "aggregate_report": report,
                "parser_failure_taxonomy": failure_taxonomy,
                "runtime_parity": "host_only_not_android_device",
            },
            "android_device": {
                "status": "not_measured",
                "gap_codes": ["android_protocol_harness_unavailable"],
            },
        },
        "gate_status": {
            "decision": "no_selection",
            "passed": False,
            "assessment": assessment,
            "additional_gap_codes": [
                "insufficient_synthetic_sample",
                "no_reproducible_quality_baseline",
                "protected_scoring_not_authorized",
            ],
        },
    }


def _walk_forbidden_result_keys(value: Any, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_RESULT_KEYS:
                raise AndroidProtocolLabError(f"{path} contains forbidden row/content field {key}")
            _walk_forbidden_result_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_forbidden_result_keys(item, f"{path}[{index}]")


def validate_result_set(value: Any, *, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate aggregate-only Phase D evidence and enforce no-selection."""

    result = _object(value, "result_set")
    _exact_keys(
        result,
        {
            "schema_version",
            "lab_id",
            "lab_version",
            "result_set_id",
            "status",
            "implementation_commit",
            "bindings",
            "privacy",
            "device_runtime_smoke",
            "profiles",
            "selection",
            "phase_e_started",
        },
        "result_set",
    )
    if (
        result["schema_version"] != 1
        or result["lab_id"] != LAB_ID
        or result["lab_version"] != LAB_VERSION
        or result["status"] != "completed_evaluation_only_no_selection"
        or not _ID_RE.fullmatch(result["result_set_id"])
        or not _COMMIT_RE.fullmatch(result["implementation_commit"])
    ):
        raise AndroidProtocolLabError("result set identity is invalid")
    if manifest is None:
        manifest = load_lab_manifest()
    bindings = _object(result["bindings"], "result_set.bindings")
    if dict(bindings) != {
        "lab_manifest_sha256": manifest["manifest_sha256"],
        "android_commit": ANDROID_COMMIT,
        "baseline_id": BASELINE_ID,
        "baseline_manifest_sha256": BASELINE_SHA256,
    }:
        raise AndroidProtocolLabError("result set bindings have drifted")
    privacy = _object(result["privacy"], "result_set.privacy")
    if dict(privacy) != {
        "classification": "aggregate_only_invented_synthetic",
        "contains_private_data": False,
        "contains_row_level_predictions": False,
        "raw_model_output_retained": False,
    }:
        raise AndroidProtocolLabError("result set privacy declaration is invalid")
    profiles = _array(result["profiles"], "result_set.profiles")
    expected_ids = {item["experiment_profile_id"] for item in manifest["profiles"]}
    actual_ids = {item.get("experiment_profile_id") for item in profiles}
    if len(profiles) != 10 or actual_ids != expected_ids:
        raise AndroidProtocolLabError("result set must contain all ten independent profiles")
    for index, profile in enumerate(profiles):
        profile_value = _object(profile, f"result_set.profiles[{index}]")
        if profile_value.get("selected_profile_id") is not None:
            raise AndroidProtocolLabError("result profile selection must remain null")
        evidence = _object(profile_value.get("evidence"), f"profiles[{index}].evidence")
        if set(evidence) != {"source_static", "host_hf", "host_gguf", "android_device"}:
            raise AndroidProtocolLabError("result profile evidence classes are incomplete")
        if evidence["android_device"].get("status") not in {"measured", "not_measured"}:
            raise AndroidProtocolLabError("Android device evidence status is invalid")
        gate = _object(profile_value.get("gate_status"), f"profiles[{index}].gate_status")
        if gate.get("decision") != "no_selection" or gate.get("passed") is not False:
            raise AndroidProtocolLabError("Phase D profile gates must remain no-selection")
    selection = _object(result["selection"], "result_set.selection")
    if dict(selection) != {
        "decision": "no_selection",
        "selected_profile_id": None,
        "direct_v2_selected": False,
        "candidate_v2_selected": False,
        "model_selected": False,
        "runtime_variant_selected": False,
        "production_defaults_changed": False,
    }:
        raise AndroidProtocolLabError("result set selection must remain null")
    if result["phase_e_started"] is not False:
        raise AndroidProtocolLabError("Phase E must not start in a Phase D result")
    _walk_forbidden_result_keys(result)
    return {**dict(result), "result_sha256": sha256_json(result)}
