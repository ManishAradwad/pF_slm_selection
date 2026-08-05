#!/usr/bin/env python3
"""Collect local-only model proposals for a blinded subset of the private pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DATA.utils import doc_to_text  # noqa: E402
from lfm25.contract import parse_prediction  # noqa: E402
from lfm25.private_data import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_jsonl,
    evaluate_consensus,
    file_sha256,
    read_jsonl,
)


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain one JSON object: {path}")
    return value


def _resolve_private_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    private_root = (repo_root / "PRIVATE_DATA" / "lfm25").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as error:
        raise ValueError("proposal outputs and manifests must remain in PRIVATE_DATA/lfm25") from error
    return resolved


def _selection_bucket(row: Mapping[str, Any]) -> tuple[str, str]:
    kind = (
        "heuristic_transaction"
        if row.get("silver_label") is not None
        else str(row.get("hard_negative_category") or "unclassified")
    )
    return str(row.get("split")), kind


def select_proposal_records(
    records: Sequence[dict[str, Any]], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Choose all heuristic positives plus deterministic stratified negatives."""

    eligible_splits = {str(value) for value in selection["eligible_splits"]}
    eligible = [row for row in records if row.get("split") in eligible_splits]
    if selection.get("test_rows_are_blinded", True) and any(
        row.get("split") == "test" for row in eligible
    ):
        raise ValueError("test rows must remain blinded from proposal models")
    maximum = int(selection["maximum_rows"])
    if maximum <= 0:
        raise ValueError("maximum proposal rows must be positive")

    positives = sorted(
        (row for row in eligible if row.get("silver_label") is not None),
        key=lambda row: str(row["record_hash"]),
    )
    selected: list[dict[str, Any]] = []
    if selection.get("include_all_heuristic_transactions", True):
        if len(positives) > maximum:
            raise ValueError("proposal maximum is smaller than the heuristic transaction pool")
        selected.extend(positives)

    selected_hashes = {str(row["record_hash"]) for row in selected}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        if str(row["record_hash"]) in selected_hashes:
            continue
        grouped[_selection_bucket(row)].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: str(row["record_hash"]))

    ordered_groups = sorted(grouped)
    offsets = Counter()
    while len(selected) < maximum:
        progressed = False
        for group in ordered_groups:
            index = offsets[group]
            values = grouped[group]
            if index >= len(values):
                continue
            selected.append(values[index])
            offsets[group] += 1
            progressed = True
            if len(selected) >= maximum:
                break
        if not progressed:
            break
    return selected


def _render_prompt(tokenizer: Any, row: Mapping[str, Any], *, thinking: bool) -> str:
    messages = [{"role": "user", "content": doc_to_text(dict(row))}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            enable_thinking=thinking,
            **kwargs,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    bos = tokenizer.bos_token
    if bos and prompt.startswith(bos):
        prompt = prompt[len(bos) :]
    return str(prompt)


def _inference_identity(
    model_config: Mapping[str, Any], inference: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], str]:
    gguf = (repo_root / str(model_config["gguf"])).resolve(strict=True)
    observed_hash = file_sha256(gguf)
    expected_hash = str(model_config["gguf_sha256"])
    if observed_hash != expected_hash:
        raise RuntimeError(f"local proposal model fingerprint mismatch: {model_config['model_id']}")
    grammar = (repo_root / str(inference["grammar"])).resolve(strict=True)
    prompt_source = repo_root / "DATA" / "utils.py"
    identity = {
        "model_id": model_config["model_id"],
        "model_family": model_config["model_family"],
        "gguf": str(gguf),
        "gguf_sha256": observed_hash,
        "tokenizer": model_config["tokenizer"],
        "tokenizer_revision": model_config["tokenizer_revision"],
        "thinking": bool(model_config.get("thinking", False)),
        "thinking_max_tokens": int(model_config.get("thinking_max_tokens", 0)),
        "thinking_repeat_penalty": float(
            model_config.get("thinking_repeat_penalty", 1.0)
        ),
        "grammar_sha256": file_sha256(grammar),
        "prompt_source_sha256": file_sha256(prompt_source),
        "n_ctx": int(inference["n_ctx"]),
        "n_gpu_layers": int(inference["n_gpu_layers"]),
        "max_tokens": int(inference["max_tokens"]),
        "temperature": float(inference["temperature"]),
        "repeat_penalty": float(inference["repeat_penalty"]),
        "seed": int(inference["seed"]),
    }
    return identity, _canonical_json_hash(identity)


def _checkpoint_path(directory: Path, model_id: str, config_hash: str) -> Path:
    slug = "".join(character if character.isalnum() else "_" for character in model_id)
    return directory / f"{slug}.{config_hash[:16]}.jsonl"


def _load_checkpoint(path: Path, config_hash: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    values: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("inference_config_hash") != config_hash:
            raise RuntimeError("proposal checkpoint contains a different inference config")
        record_hash = str(row.get("record_hash", ""))
        proposal = row.get("proposal")
        if not record_hash or not isinstance(proposal, dict):
            raise RuntimeError("proposal checkpoint contains an invalid row")
        values[record_hash] = proposal
    return values


def _append_checkpoint(
    handle: Any,
    *,
    record_hash: str,
    proposal: Mapping[str, Any],
    config_hash: str,
) -> None:
    handle.write(
        json.dumps(
            {
                "record_hash": record_hash,
                "inference_config_hash": config_hash,
                "proposal": proposal,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def _generate_prediction(
    model: Any,
    tokenizer: Any,
    grammar: Any,
    row: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    inference: Mapping[str, Any],
) -> str:
    thinking = bool(model_config.get("thinking", False))
    prompt = _render_prompt(tokenizer, row, thinking=thinking)
    stop = ["</s>", "<|im_end|>", "<|endoftext|>", "<end_of_turn>"]
    if thinking:
        phase_one_prompt = prompt + "<think>\n"
        phase_one = model.create_completion(
            prompt=phase_one_prompt,
            temperature=float(inference["temperature"]),
            max_tokens=int(model_config["thinking_max_tokens"]),
            stop=["</think>"],
            echo=False,
            repeat_penalty=float(model_config["thinking_repeat_penalty"]),
        )
        reasoning = str(phase_one["choices"][0]["text"])
        prompt = phase_one_prompt + reasoning + "</think>\n"
    response = model.create_completion(
        prompt=prompt,
        temperature=float(inference["temperature"]),
        max_tokens=int(inference["max_tokens"]),
        stop=stop,
        echo=False,
        grammar=grammar,
        repeat_penalty=float(inference["repeat_penalty"]),
    )
    return str(response["choices"][0]["text"])


def _run_model(
    selected: Sequence[dict[str, Any]],
    *,
    model_config: Mapping[str, Any],
    inference: Mapping[str, Any],
    checkpoint_dir: Path,
    repo_root: Path,
    force: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from llama_cpp import Llama, LlamaGrammar
    from transformers import AutoTokenizer

    identity, config_hash = _inference_identity(model_config, inference, repo_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.chmod(0o700)
    checkpoint = _checkpoint_path(checkpoint_dir, str(model_config["model_id"]), config_hash)
    if force and checkpoint.exists():
        checkpoint.unlink()
    completed = _load_checkpoint(checkpoint, config_hash)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["tokenizer"]),
        revision=str(model_config["tokenizer_revision"]),
        local_files_only=True,
    )
    grammar = LlamaGrammar.from_file(str((repo_root / str(inference["grammar"])).resolve()))
    model = Llama(
        model_path=identity["gguf"],
        n_ctx=int(inference["n_ctx"]),
        n_gpu_layers=int(inference["n_gpu_layers"]),
        seed=int(inference["seed"]),
        verbose=False,
    )
    mode = "a" if checkpoint.exists() else "w"
    with checkpoint.open(mode, encoding="utf-8", newline="\n") as handle:
        os.chmod(checkpoint, 0o600)
        for index, row in enumerate(selected, start=1):
            record_hash = str(row["record_hash"])
            if record_hash in completed:
                continue
            raw_prediction = _generate_prediction(
                model,
                tokenizer,
                grammar,
                row,
                model_config=model_config,
                inference=inference,
            )
            parsed = parse_prediction(raw_prediction)
            valid = parsed.status in {"null", "transaction"}
            proposal = {
                "model_id": model_config["model_id"],
                "model_family": model_config["model_family"],
                "label": parsed.value if valid else None,
                "confidence": float(
                    inference["confidence_for_schema_valid_greedy_output"]
                ) if valid else 0.0,
                "confidence_basis": inference["confidence_basis"],
                "inference_config_hash": config_hash,
                "schema_valid": valid,
                "raw_output_persisted": False,
            }
            completed[record_hash] = proposal
            _append_checkpoint(
                handle,
                record_hash=record_hash,
                proposal=proposal,
                config_hash=config_hash,
            )
            if index % 25 == 0 or index == len(selected):
                print(
                    json.dumps(
                        {
                            "model_id": model_config["model_id"],
                            "completed_rows": len(completed),
                            "selected_rows": len(selected),
                        },
                        sort_keys=True,
                    )
                )
    del model
    return completed, {**identity, "inference_config_hash": config_hash}


def _proposal_report(
    records: Sequence[Mapping[str, Any]],
    selected_hashes: set[str],
    identities: Sequence[Mapping[str, Any]],
    manifest_before_sha256: str,
) -> dict[str, Any]:
    selected = [row for row in records if str(row["record_hash"]) in selected_hashes]
    consensus_status = Counter(
        str(row.get("consensus_acceptance", {}).get("status")) for row in selected
    )
    accepted_kind = Counter()
    for row in selected:
        consensus = row.get("consensus_acceptance", {})
        if consensus.get("accepted"):
            accepted_kind[
                "null" if consensus.get("accepted_label") is None else "transaction"
            ] += 1
    return {
        "schema_version": 1,
        "dataset_state": "private_local_only",
        "manifest_before_sha256": manifest_before_sha256,
        "selected_rows": len(selected),
        "unselected_rows": len(records) - len(selected),
        "test_rows_proposed": sum(row.get("split") == "test" for row in selected),
        "proposal_count": sum(len(row.get("local_model_proposals", [])) for row in selected),
        "schema_valid_proposal_count": sum(
            proposal.get("schema_valid") is True
            for row in selected
            for proposal in row.get("local_model_proposals", [])
        ),
        "consensus_status_counts": dict(sorted(consensus_status.items())),
        "consensus_accepted_label_counts": dict(sorted(accepted_kind.items())),
        "models": list(identities),
        "privacy": {
            "raw_sms_logged": False,
            "raw_model_outputs_persisted": False,
            "outputs_remain_gitignored": True,
            "hosted_inference_used": False,
        },
    }


def _review_reasons(row: Mapping[str, Any], selected_hashes: set[str]) -> list[str]:
    reasons: list[str] = []
    if row.get("split") == "test":
        reasons.append("human_gold_test")
    if str(row.get("record_hash")) in selected_hashes:
        consensus = row.get("consensus_acceptance", {})
        if not consensus.get("accepted"):
            reasons.append("local_model_disagreement_or_insufficient_consensus")
    if float(row.get("confidence", 0.0)) < 0.9:
        reasons.append("low_heuristic_confidence")
    if row.get("hard_negative_category") is not None:
        reasons.append("hard_negative")
    return reasons or ["routine_review"]


def _merge_proposals(
    records: list[dict[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    proposals_by_model: Sequence[Mapping[str, Mapping[str, Any]]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_hashes = {str(row["record_hash"]) for row in selected}
    for row in records:
        record_hash = str(row["record_hash"])
        if record_hash not in selected_hashes:
            row["local_model_proposals"] = []
            row["consensus_acceptance"] = {
                **row.get("consensus_acceptance", {}),
                "status": "not_selected_for_local_proposals",
                "accepted": False,
                "accepted_label": None,
            }
            continue
        row["local_model_proposals"] = [
            values[record_hash]
            for values in proposals_by_model
            if record_hash in values
        ]
        row["consensus_acceptance"] = evaluate_consensus(row, policy)

    queue: list[dict[str, Any]] = []
    for row in records:
        value = dict(row)
        reasons = _review_reasons(row, selected_hashes)
        value["review_reason_codes"] = reasons
        value["review_priority"] = (
            0
            if "human_gold_test" in reasons
            else 1
            if "local_model_disagreement_or_insufficient_consensus" in reasons
            else 2
            if "hard_negative" in reasons
            else 3
        )
        queue.append(value)
    records.sort(key=lambda row: str(row["record_hash"]))
    queue.sort(key=lambda row: (row["review_priority"], str(row["record_hash"])))
    return records, queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "lfm25" / "private_proposals.json",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=REPO_ROOT / "configs" / "lfm25_data.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _load_json_object(args.config)
    data_config = _load_json_object(args.data_config)
    manifest = _resolve_private_path(config["private_manifest"], repo_root=REPO_ROOT)
    queue_path = _resolve_private_path(config["review_queue"], repo_root=REPO_ROOT)
    report_path = _resolve_private_path(config["report"], repo_root=REPO_ROOT)
    checkpoint_dir = _resolve_private_path(config["checkpoint_dir"], repo_root=REPO_ROOT)
    manifest_before_sha256 = file_sha256(manifest)
    records = read_jsonl(manifest)
    selected = select_proposal_records(records, config["selection"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected_rows": len(selected),
                    "selected_buckets": dict(
                        sorted(Counter(str(_selection_bucket(row)) for row in selected).items())
                    ),
                    "test_rows_selected": sum(row.get("split") == "test" for row in selected),
                },
                sort_keys=True,
            )
        )
        return 0

    proposals_by_model: list[dict[str, dict[str, Any]]] = []
    identities: list[dict[str, Any]] = []
    for model_config in config["models"]:
        proposals, identity = _run_model(
            selected,
            model_config=model_config,
            inference=config["inference"],
            checkpoint_dir=checkpoint_dir,
            repo_root=REPO_ROOT,
            force=args.force,
        )
        proposals_by_model.append(proposals)
        identities.append(identity)

    records, queue = _merge_proposals(
        records,
        selected,
        proposals_by_model,
        data_config["consensus_policy"],
    )
    _atomic_write_jsonl(manifest, records)
    _atomic_write_jsonl(queue_path, queue)
    report = _proposal_report(
        records,
        {str(row["record_hash"]) for row in selected},
        identities,
        manifest_before_sha256,
    )
    report["manifest_after_sha256"] = file_sha256(manifest)
    _atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "selected_rows": report["selected_rows"],
                "proposal_count": report["proposal_count"],
                "consensus_status_counts": report["consensus_status_counts"],
                "consensus_accepted_label_counts": report[
                    "consensus_accepted_label_counts"
                ],
                "test_rows_proposed": report["test_rows_proposed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
