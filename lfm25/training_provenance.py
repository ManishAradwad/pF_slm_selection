"""Private-safe, adapter-bound training-run identities."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from lfm25.provenance import file_sha256


RUN_MANIFEST_FORMAT = "lfm25_lora_run_manifest_v2"
ADAPTER_ARTIFACT_FORMAT = "peft_adapter_artifact_v1"
ARTIFACT_BINDING_FORMAT = "lfm25_training_adapter_binding_v1"

LORA_TARGETS = (
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
)
_COMMON_TRAINER_CODE_PATHS = (
    "scripts/train_lfm25_lora.py",
    "lfm25/training_provenance.py",
    "lfm25/training_loss.py",
    "lfm25/contract.py",
)
TRAINER_CODE_PATHS_BY_ARM = {
    "direct": (
        *_COMMON_TRAINER_CODE_PATHS,
        "lfm25/prompts.py",
        "lfm25/android_contract.py",
    ),
    "selector": (
        *_COMMON_TRAINER_CODE_PATHS,
        "scripts/train_lfm25_candidate_protocol_v1.py",
        "lfm25/candidate_protocol.py",
        "lfm25/candidates.py",
    ),
}

_BASE_MODEL_FIXED_FILENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_BASE_MODEL_WEIGHT_RE = re.compile(
    r"(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)"
    r"(?:\.index\.json)?"
)
_ADAPTER_WEIGHT_RE = re.compile(
    r"adapter_model(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)"
    r"(?:\.index\.json)?"
)
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _json_identity(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"run_manifest.json has invalid {field}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"run_manifest.json has invalid {field}")
    return value


def _controlled_text(value: Any, field: str, allowed: set[str]) -> str:
    text = _text(value, field)
    if text not in allowed:
        raise ValueError(f"run_manifest.json has invalid {field}")
    return text


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"run_manifest.json has invalid {field}")
    return text


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"run_manifest.json has invalid {field}")
    if minimum is not None and value < minimum:
        raise ValueError(f"run_manifest.json has invalid {field}")
    return value


def _number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"run_manifest.json has invalid {field}")
    if not math.isfinite(float(value)):
        raise ValueError(f"run_manifest.json has invalid {field}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"run_manifest.json has invalid {field}")
    return value


def _safe_base_model_filename(name: Any) -> str:
    if not isinstance(name, str) or (
        name not in _BASE_MODEL_FIXED_FILENAMES and _BASE_MODEL_WEIGHT_RE.fullmatch(name) is None
    ):
        raise ValueError("run_manifest.json has invalid base model filename")
    return name


def _safe_adapter_filename(name: Any) -> str:
    if not isinstance(name, str) or (
        name != "adapter_config.json" and _ADAPTER_WEIGHT_RE.fullmatch(name) is None
    ):
        raise ValueError("run_manifest.json has invalid adapter artifact filename")
    return name


def _file_evidence(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def adapter_artifact_evidence(adapter: Path) -> dict[str, Any]:
    """Fingerprint exactly the PEFT config and weights loaded from an adapter."""

    root = Path(adapter).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("adapter path is not a directory")
    names = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and (path.name == "adapter_config.json" or _ADAPTER_WEIGHT_RE.fullmatch(path.name))
    }
    weight_names = {
        name
        for name in names
        if _ADAPTER_WEIGHT_RE.fullmatch(name) and not name.endswith(".index.json")
    }
    if "adapter_config.json" not in names or not weight_names:
        raise ValueError("adapter artifact is missing its config or weights")
    files = {name: _file_evidence(root / name) for name in sorted(names)}
    identity_payload = {"format": ADAPTER_ARTIFACT_FORMAT, "files": files}
    return {**identity_payload, "identity_sha256": _json_identity(identity_payload)}


def trainer_code_fingerprints(repo_root: Path, arm: str) -> dict[str, str]:
    """Return the exact trainer-side code identity for a controlled arm."""

    try:
        names = TRAINER_CODE_PATHS_BY_ARM[arm]
    except KeyError as error:
        raise ValueError(f"unknown training arm: {arm!r}") from error
    root = Path(repo_root).resolve(strict=True)
    return {name: file_sha256(root / name) for name in names}


def _training_arm(prompt_profile: str) -> str:
    if prompt_profile in {"candidate_protocol_v1", "candidate_selector"}:
        return "selector"
    return "direct"


def _safe_fingerprint(value: Any, field: str, *, filename: str) -> dict[str, Any]:
    item = _mapping(value, field)
    if item.get("filename") != filename:
        raise ValueError(f"run_manifest.json has invalid {field}.filename")
    return {
        "filename": filename,
        "bytes": _integer(item.get("bytes"), f"{field}.bytes", minimum=1),
        "sha256": _sha256(item.get("sha256"), f"{field}.sha256"),
    }


def _safe_adapter_artifact(value: Any) -> dict[str, Any]:
    artifact = _mapping(value, "adapter_artifact")
    if artifact.get("format") != ADAPTER_ARTIFACT_FORMAT:
        raise ValueError("run_manifest.json has invalid adapter_artifact.format")
    files_value = _mapping(artifact.get("files"), "adapter_artifact.files")
    files: dict[str, dict[str, Any]] = {}
    for raw_name, raw_evidence in sorted(files_value.items()):
        name = _safe_adapter_filename(raw_name)
        item = _mapping(raw_evidence, f"adapter_artifact.files.{name}")
        files[name] = {
            "bytes": _integer(
                item.get("bytes"),
                f"adapter_artifact.files.{name}.bytes",
                minimum=1,
            ),
            "sha256": _sha256(
                item.get("sha256"),
                f"adapter_artifact.files.{name}.sha256",
            ),
        }
    weight_names = {
        name
        for name in files
        if _ADAPTER_WEIGHT_RE.fullmatch(name) and not name.endswith(".index.json")
    }
    if "adapter_config.json" not in files or not weight_names:
        raise ValueError("run_manifest.json has incomplete adapter_artifact.files")
    identity_payload = {"format": ADAPTER_ARTIFACT_FORMAT, "files": files}
    identity = _sha256(
        artifact.get("identity_sha256"),
        "adapter_artifact.identity_sha256",
    )
    if identity != _json_identity(identity_payload):
        raise ValueError("run_manifest.json has inconsistent adapter_artifact identity")
    return {**identity_payload, "identity_sha256": identity}


def _contract_identity(
    contract: Mapping[str, Any],
    prompt_profile: str,
) -> tuple[str, str, int]:
    if prompt_profile == "android":
        expected = ("pocketfinancer", "pocketfinancer", 3)
        observed = (
            contract.get("name"),
            contract.get("profile"),
            contract.get("version"),
        )
    elif prompt_profile == "candidate_protocol_v1":
        expected = ("candidate_protocol_v1", "candidate_protocol_v1", 1)
        observed = (
            contract.get("name"),
            contract.get("profile"),
            contract.get("version"),
        )
    elif prompt_profile == "legacy":
        expected = ("lfm25_short_extraction", "legacy", 1)
        observed = (
            contract.get("contract"),
            contract.get("profile"),
            contract.get("contract_version"),
        )
    elif prompt_profile == "candidate_selector":
        expected = ("grounded_candidate_selector", "candidate_selector", 1)
        observed = (
            contract.get("contract"),
            contract.get("profile"),
            contract.get("contract_version", 1),
        )
    else:
        raise ValueError("run_manifest.json has invalid optimization.prompt_profile")
    if observed != expected:
        raise ValueError("run_manifest.json has invalid contract identity")
    return expected


def _checkpoint_selection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    state = _mapping(manifest.get("trainer_state"), "trainer_state")
    load_best = _boolean(
        state.get("load_best_model_at_end"),
        "trainer_state.load_best_model_at_end",
    )
    restored = _boolean(
        state.get("load_best_model_at_end_restored_best_checkpoint"),
        "trainer_state.load_best_model_at_end_restored_best_checkpoint",
    )
    if not load_best or not restored:
        raise ValueError("run_manifest.json did not restore the selected best checkpoint")

    best_path = _text(
        state.get("best_model_checkpoint"),
        "trainer_state.best_model_checkpoint",
    )
    restored_path = _text(
        state.get("restored_best_model_checkpoint"),
        "trainer_state.restored_best_model_checkpoint",
    )
    best_match = _CHECKPOINT_RE.fullmatch(Path(best_path).name)
    restored_match = _CHECKPOINT_RE.fullmatch(Path(restored_path).name)
    if (
        best_match is None
        or restored_match is None
        or best_match.group(1) != restored_match.group(1)
    ):
        raise ValueError("run_manifest.json has inconsistent best-checkpoint paths")
    best_step = int(best_match.group(1))

    best_log = _mapping(state.get("best_eval_log"), "trainer_state.best_eval_log")
    metric_name = _controlled_text(
        best_log.get("metric_name"),
        "trainer_state.best_eval_log.metric_name",
        {"eval_loss"},
    )
    metric_value = _number(
        best_log.get("metric_value"),
        "trainer_state.best_eval_log.metric_value",
    )
    best_metric = _number(state.get("best_metric"), "trainer_state.best_metric")
    if not math.isclose(
        float(metric_value),
        float(best_metric),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("run_manifest.json has inconsistent best metric")
    log_step = _integer(
        best_log.get("step"),
        "trainer_state.best_eval_log.step",
        minimum=1,
    )
    if log_step != best_step:
        raise ValueError("run_manifest.json has inconsistent best checkpoint step")
    final_step = _integer(
        state.get("final_global_step"),
        "trainer_state.final_global_step",
        minimum=1,
    )
    if best_step > final_step:
        raise ValueError("run_manifest.json has impossible best checkpoint step")
    return {
        "best_step": best_step,
        "best_metric_name": metric_name,
        "best_metric_value": metric_value,
        "best_epoch": _number(
            best_log.get("epoch"),
            "trainer_state.best_eval_log.epoch",
        ),
        "final_global_step": final_step,
        "final_epoch": _number(
            state.get("final_epoch"),
            "trainer_state.final_epoch",
        ),
        "load_best_model_at_end": True,
        "restored_best_checkpoint": True,
    }


def _safe_training_manifest_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only aggregate, controlled-run identity from a v2 trainer manifest."""

    if manifest.get("manifest_format") != RUN_MANIFEST_FORMAT:
        raise ValueError("run_manifest.json has an unsupported manifest_format")

    contract = _mapping(manifest.get("contract"), "contract")
    loss = _mapping(manifest.get("loss"), "loss")
    weights = _mapping(loss.get("weights"), "loss.weights")
    lora = _mapping(manifest.get("lora"), "lora")
    optimization = _mapping(manifest.get("optimization"), "optimization")
    base_model = _mapping(manifest.get("base_model_provenance"), "base_model_provenance")
    base_files = _mapping(base_model.get("files"), "base_model_provenance.files")
    safe_base_files: dict[str, str] = {}
    for raw_name, raw_evidence in sorted(base_files.items()):
        name = _safe_base_model_filename(raw_name)
        item = _mapping(raw_evidence, f"base_model_provenance.files.{name}")
        safe_base_files[name] = _sha256(
            item.get("sha256"),
            f"base_model_provenance.files.{name}.sha256",
        )
    if not safe_base_files:
        raise ValueError("run_manifest.json has no base model fingerprints")
    if (
        "config.json" not in safe_base_files
        or not ({"tokenizer.json", "tokenizer.model"} & set(safe_base_files))
        or not any(name.endswith((".safetensors", ".bin")) for name in safe_base_files)
    ):
        raise ValueError("run_manifest.json has incomplete base model fingerprints")

    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or tuple(target_modules) != LORA_TARGETS:
        raise ValueError("run_manifest.json has invalid lora.target_modules")

    prompt_profile = _controlled_text(
        optimization.get("prompt_profile"),
        "optimization.prompt_profile",
        {"android", "candidate_protocol_v1", "candidate_selector", "legacy"},
    )
    contract_name, contract_profile, contract_version = _contract_identity(
        contract,
        prompt_profile,
    )
    prompt = {
        "profile": prompt_profile,
        "contract_name": contract_name,
        "contract_profile": contract_profile,
        "contract_version": contract_version,
        "contract_sha256": _json_identity(contract),
    }
    safe_loss = {
        "mode": _controlled_text(
            loss.get("mode"),
            "loss.mode",
            {"per_example_completion_mean"},
        ),
        "causal_shift": _boolean(loss.get("causal_shift"), "loss.causal_shift"),
        "ignore_index": _integer(loss.get("ignore_index"), "loss.ignore_index"),
        "token_reduction": _controlled_text(
            loss.get("token_reduction"),
            "loss.token_reduction",
            {"weighted_mean_per_example"},
        ),
        "example_reduction": _controlled_text(
            loss.get("example_reduction"),
            "loss.example_reduction",
            {"sample_weighted_mean"},
        ),
        "first_supervised_token_weight": _number(
            weights.get("first_supervised_token_weight"),
            "loss.weights.first_supervised_token_weight",
        ),
    }
    if safe_loss["causal_shift"] is not True or safe_loss["ignore_index"] != -100:
        raise ValueError("run_manifest.json has invalid completion-only loss semantics")

    safe_lora = {
        "rank": _integer(lora.get("rank"), "lora.rank", minimum=1),
        "alpha": _integer(lora.get("alpha"), "lora.alpha", minimum=1),
        "dropout": _number(lora.get("dropout"), "lora.dropout"),
        "target_modules": list(LORA_TARGETS),
    }
    if not 0 <= float(safe_lora["dropout"]) < 1:
        raise ValueError("run_manifest.json has invalid lora.dropout")

    numeric_optimization = (
        "learning_rate",
        "epochs_requested",
        "batch_size",
        "gradient_accumulation",
        "effective_batch_size",
        "max_length",
        "first_supervised_token_weight",
        "warmup_ratio",
        "warmup_steps",
        "weight_decay",
        "early_stopping_patience",
        "max_grad_norm",
        "per_device_eval_batch_size",
    )
    safe_optimization: dict[str, Any] = {
        key: _number(optimization.get(key), f"optimization.{key}") for key in numeric_optimization
    }
    text_allowed = {
        "loss_mode": {"per_example_completion_mean"},
        "optimizer": {"adamw_torch"},
        "lr_scheduler_type": {"cosine"},
        "eval_strategy": {"epoch"},
        "save_strategy": {"epoch"},
    }
    safe_optimization.update(
        {
            key: _controlled_text(
                optimization.get(key),
                f"optimization.{key}",
                allowed,
            )
            for key, allowed in text_allowed.items()
        }
    )
    safe_optimization.update(
        {
            key: _boolean(optimization.get(key), f"optimization.{key}")
            for key in (
                "bf16",
                "tf32",
                "gradient_checkpointing",
                "full_determinism",
            )
        }
    )
    use_reentrant = _boolean(
        optimization.get("gradient_checkpointing_use_reentrant"),
        "optimization.gradient_checkpointing_use_reentrant",
    )
    if use_reentrant:
        raise ValueError(
            "run_manifest.json has invalid optimization.gradient_checkpointing_use_reentrant"
        )
    safe_optimization["gradient_checkpointing_use_reentrant"] = False

    datasets = {
        "train_sha256": _sha256(manifest.get("train_sha256"), "train_sha256"),
        "eval_sha256": _sha256(manifest.get("eval_sha256"), "eval_sha256"),
    }
    report_value = manifest.get("dataset_report")
    if report_value is not None:
        datasets["report"] = _safe_fingerprint(
            report_value,
            "dataset_report",
            filename="candidate_protocol_v1_report.json",
        )

    code_value = _mapping(manifest.get("trainer_code_sha256"), "trainer_code_sha256")
    expected_code_names = TRAINER_CODE_PATHS_BY_ARM[_training_arm(prompt_profile)]
    if set(code_value) != set(expected_code_names):
        raise ValueError("run_manifest.json has invalid trainer_code_sha256 keys")
    trainer_code = {
        name: _sha256(code_value.get(name), f"trainer_code_sha256.{name}")
        for name in expected_code_names
    }

    evidence = {
        "seed": _integer(manifest.get("seed"), "seed", minimum=0),
        "datasets": datasets,
        "base_model": {
            "files": safe_base_files,
            "identity_sha256": _json_identity(safe_base_files),
        },
        "prompt": prompt,
        "loss": safe_loss,
        "lora": safe_lora,
        "optimization": safe_optimization,
    }
    evidence["identity_sha256"] = _json_identity(evidence)
    evidence.update(
        {
            "model_lock": _safe_fingerprint(
                manifest.get("model_lock"),
                "model_lock",
                filename="model.lock.json",
            ),
            "adapter_artifact": _safe_adapter_artifact(manifest.get("adapter_artifact")),
            "checkpoint_selection": _checkpoint_selection(manifest),
            "trainer_code_sha256": trainer_code,
        }
    )
    return evidence


def _artifact_binding_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": ARTIFACT_BINDING_FORMAT,
        "training_identity_sha256": evidence["identity_sha256"],
        "model_lock": evidence["model_lock"],
        "adapter_artifact_identity_sha256": evidence["adapter_artifact"]["identity_sha256"],
        "checkpoint_selection": evidence["checkpoint_selection"],
        "trainer_code_sha256": evidence["trainer_code_sha256"],
    }


def training_manifest_artifact_binding(manifest: Mapping[str, Any]) -> str:
    """Hash the manifest identity and adapter evidence without self-reference."""

    evidence = _safe_training_manifest_payload(manifest)
    return _json_identity(_artifact_binding_payload(evidence))


def _safe_training_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _safe_training_manifest_payload(manifest)
    stored_binding = _sha256(
        manifest.get("artifact_binding_sha256"),
        "artifact_binding_sha256",
    )
    expected_binding = _json_identity(_artifact_binding_payload(evidence))
    if not hmac.compare_digest(stored_binding, expected_binding):
        raise ValueError("run_manifest.json has inconsistent artifact binding")
    return {**evidence, "artifact_binding_sha256": stored_binding}


def training_run_manifest_evidence(adapter: Path | None) -> dict[str, Any] | None:
    """Return private-safe, adapter-bound run evidence when a manifest exists.

    Historical adapters without a manifest remain evaluable. A present manifest
    must use the cryptographically bound v2 schema; older unbound manifests cannot
    honestly establish controlled-comparison provenance and therefore fail closed.
    """

    if adapter is None:
        return None
    adapter_root = Path(adapter).resolve(strict=True)
    manifest_path = adapter_root.parent / "run_manifest.json"
    if not manifest_path.is_file():
        return None

    manifest_bytes = manifest_path.read_bytes()
    fingerprint = {
        "filename": "run_manifest.json",
        "bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    try:
        manifest = json.loads(manifest_bytes)
        safe = _safe_training_manifest(_mapping(manifest, "root"))
        actual_adapter = adapter_artifact_evidence(adapter_root)
        if safe["adapter_artifact"] != actual_adapter:
            raise ValueError("run_manifest.json does not describe the evaluated adapter")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return {
            "present": True,
            "valid": False,
            "manifest": fingerprint,
            "error": type(error).__name__,
        }
    return {
        "present": True,
        "valid": True,
        "manifest": fingerprint,
        **safe,
    }
