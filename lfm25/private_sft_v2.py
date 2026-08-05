"""Privacy-preserving v2 SFT materialization from the local split manifest.

Only rows already assigned to the original ``train`` split are considered.  The
original dev and test partitions remain sealed from filtering, label selection,
sampling, and the derived train/dev artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .private_data import (
    PrivateDataError,
    _DEBIT_RE,
    _CREDIT_RE,
    _atomic_write_json,
    _atomic_write_jsonl,
    _parse_timestamp,
    _sender_template_components,
    canonical_exact_text,
    canonical_label,
    ensure_within,
    file_sha256,
    normalize_unicode,
    read_jsonl,
    require_private_ignore,
)

try:
    from .android_contract import (
        contract_provenance as _android_contract_provenance,
        prefilter_sms as _android_prefilter_sms,
    )
except ImportError:  # Fail closed at build time if the optional contract is unavailable.
    _android_contract_provenance = None
    _android_prefilter_sms = None


BUILDER_VERSION = "lfm25-private-sft-v2.1"
OUTPUT_SCHEMA_VERSION = 2
DEFAULT_MANIFEST_NAME = "split_manifest.jsonl"
OUTPUT_FILENAMES = {
    "train": "private_sft_v2_train.jsonl",
    "dev": "private_sft_v2_dev.jsonl",
    "metadata": "private_sft_v2_metadata.json",
    "report": "private_sft_v2_report.json",
}
LABEL_TIERS = ("human_gold", "consensus_silver", "grounded_silver")
TIER_PRIORITY = {tier: index for index, tier in enumerate(LABEL_TIERS)}
MINIMUM_CONSENSUS_VALID_PROPOSALS = 3
MINIMUM_CONSENSUS_AGREEMENT = 2
MINIMUM_CONSENSUS_FAMILIES = 2
MINIMUM_CONSENSUS_CONFIDENCE = 0.9
_COUNTERPARTY_CURRENCY_AMOUNT_RE = re.compile(
    r"(?i)(?:\b(?:inr|rs)\.?\s*[:=\-]?\s*\d|\u20b9\s*\d)"
)


class PrivateSFTV2Error(PrivateDataError):
    """A v2 builder failure whose message contains no private row values."""


@dataclass(frozen=True)
class BuildConfig:
    """Deterministic selection and split policy."""

    dev_fraction: float = 0.15
    seed: int = 25_052_027
    minimum_silver_confidence: float = 0.86
    max_per_template: int = 8
    max_per_category: int = 512
    max_null_to_transaction_ratio: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.dev_fraction < 1.0:
            raise PrivateSFTV2Error("dev fraction must be strictly between zero and one")
        if not 0.0 <= self.minimum_silver_confidence <= 1.0:
            raise PrivateSFTV2Error("minimum silver confidence must be in [0, 1]")
        if self.max_per_template < 1 or self.max_per_category < 1:
            raise PrivateSFTV2Error("template and category caps must be positive")
        if not math.isfinite(self.max_null_to_transaction_ratio):
            raise PrivateSFTV2Error("null-to-transaction ratio must be finite")
        if self.max_null_to_transaction_ratio < 0:
            raise PrivateSFTV2Error("null-to-transaction ratio cannot be negative")

    def public_dict(self) -> dict[str, Any]:
        return {
            "dev_fraction": self.dev_fraction,
            "seed": self.seed,
            "minimum_silver_confidence": self.minimum_silver_confidence,
            "max_per_template": self.max_per_template,
            "max_per_category": self.max_per_category,
            "max_null_to_transaction_ratio": self.max_null_to_transaction_ratio,
        }


@dataclass
class BuildResult:
    train_rows: list[dict[str, Any]]
    dev_rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    report: dict[str, Any]


@dataclass(frozen=True)
class _LabelChoice:
    expected: Any
    tier: str
    confidence: float
    sample_weight: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _Candidate:
    record: Mapping[str, Any]
    choice: _LabelChoice
    category: str

    @property
    def record_hash(self) -> str:
        return str(self.record["record_hash"])

    @property
    def template_group(self) -> str:
        return str(self.record["template_group"])

    @property
    def sender_hash(self) -> str:
        return str(self.record["private_hashes"]["sender"])


_SOURCE_AMOUNT_RE = re.compile(
    r"(?:₹|\brs\.?|\binr\b)\s*[-+]?\s*(?P<prefix>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<suffix>\d[\d,]*(?:\.\d+)?)\s*(?:inr\b|rupees?\b)",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NON_ACCOUNT_RE = re.compile(r"[^a-z0-9*x]+")
_SAFE_REASON_RE = re.compile(r"[^a-z0-9_.-]+")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _config_sha256(config: BuildConfig) -> str:
    return hashlib.sha256(_canonical_json(config.public_dict()).encode("utf-8")).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_hash_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    joined = "\n".join(sorted(str(row["source"]["record_hash"]) for row in rows)) + "\n"
    return _sha256_bytes(joined.encode("utf-8"))


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        return None
    return converted


def _amounts_in_source(sms: str) -> tuple[float, ...]:
    values: list[float] = []
    for match in _SOURCE_AMOUNT_RE.finditer(sms):
        raw = match.group("prefix") or match.group("suffix")
        try:
            amount = float(raw.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(amount) and amount >= 0:
            values.append(amount)
    return tuple(values)


def _account_fingerprint(value: str) -> str:
    normalized = normalize_unicode(value).casefold().replace("*", "x")
    return _NON_ACCOUNT_RE.sub("", normalized)


def _phrase_fingerprint(value: str) -> str:
    return " ".join(_NON_ALNUM_RE.sub(" ", canonical_exact_text(value)).split())


def grounding_errors(sms: str, label: Any) -> tuple[str, ...]:
    """Return aggregate-safe reasons when a transaction label is not source-grounded."""

    try:
        expected = canonical_label(label)
    except PrivateDataError:
        return ("invalid_schema",)
    if expected is None:
        return ()

    errors: list[str] = []
    source_amounts = _amounts_in_source(sms)
    if not any(
        math.isclose(float(expected["amount"]), value, abs_tol=0.005)
        for value in source_amounts
    ):
        errors.append("amount_not_grounded")

    account = _account_fingerprint(expected["account"])
    source_account = _account_fingerprint(sms)
    if (
        len(account) < 4
        or not any(value.isdigit() for value in account)
        or account not in source_account
    ):
        errors.append("account_not_grounded")

    counterparty = expected["counterparty"]
    if counterparty is not None:
        party = _phrase_fingerprint(counterparty)
        source_phrase = _phrase_fingerprint(sms)
        if len(party) < 2 or party not in source_phrase:
            errors.append("counterparty_not_grounded")
        if _COUNTERPARTY_CURRENCY_AMOUNT_RE.search(str(counterparty)):
            errors.append("counterparty_contains_currency_amount")

    if expected["type"] == "debit" and not _DEBIT_RE.search(sms):
        errors.append("type_not_grounded")
    if expected["type"] == "credit" and not _CREDIT_RE.search(sms):
        errors.append("type_not_grounded")
    return tuple(errors)


def _matching_consensus_evidence(
    record: Mapping[str, Any], label: Any
) -> list[tuple[float, str]]:
    expected = _canonical_json(canonical_label(label))
    evidence: list[tuple[float, str]] = []
    seen_models: set[str] = set()
    proposals = record.get("local_model_proposals")
    if not isinstance(proposals, list):
        return evidence
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        model_id = proposal.get("model_id")
        model_family = proposal.get("model_family")
        if (
            not isinstance(model_id, str)
            or not model_id
            or model_id in seen_models
            or not isinstance(model_family, str)
            or not model_family
        ):
            continue
        try:
            proposal_label = canonical_label(proposal.get("label"))
        except PrivateDataError:
            continue
        proposal_confidence = _confidence(proposal.get("confidence"))
        if proposal_confidence is None or _canonical_json(proposal_label) != expected:
            continue
        seen_models.add(model_id)
        evidence.append((proposal_confidence, model_family))
    return evidence


def _human_choice(record: Mapping[str, Any]) -> tuple[_LabelChoice | None, str | None]:
    if record.get("review_status") != "human_approved":
        return None, None
    reviewer = record.get("human_reviewer")
    reviewed_at = record.get("human_reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        return None, "human_missing_reviewer"
    if not isinstance(reviewed_at, str) or _parse_timestamp(reviewed_at) is None:
        return None, "human_missing_review_time"
    if "human_label" not in record:
        return None, "human_invalid_label"
    try:
        label = canonical_label(record.get("human_label"))
    except PrivateDataError:
        return None, "human_invalid_label"
    return (
        _LabelChoice(
            expected=label,
            tier="human_gold",
            confidence=1.0,
            sample_weight=1.0,
            provenance={
                "label_source": "human_approved_manifest_label",
                "human_reviewer": reviewer,
                "human_reviewed_at": reviewed_at,
                "gold": True,
            },
        ),
        None,
    )


def _consensus_choice(record: Mapping[str, Any]) -> tuple[_LabelChoice | None, str | None]:
    consensus = record.get("consensus_acceptance")
    if not isinstance(consensus, Mapping) or consensus.get("accepted") is not True:
        return None, None
    if consensus.get("status") != "accepted":
        return None, "consensus_status_mismatch"
    counts = (
        consensus.get("valid_proposal_count"),
        consensus.get("agreeing_model_count"),
        consensus.get("independent_model_family_count"),
    )
    required_counts = (
        MINIMUM_CONSENSUS_VALID_PROPOSALS,
        MINIMUM_CONSENSUS_AGREEMENT,
        MINIMUM_CONSENSUS_FAMILIES,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < required
        for value, required in zip(counts, required_counts, strict=True)
    ):
        return None, "consensus_invalid_counts"
    if "accepted_label" not in consensus:
        return None, "consensus_invalid_label"
    try:
        label = canonical_label(consensus.get("accepted_label"))
    except PrivateDataError:
        return None, "consensus_invalid_label"
    evidence = _matching_consensus_evidence(record, label)
    if (
        len(evidence) < int(counts[1])
        or len({family for _, family in evidence}) < int(counts[2])
    ):
        return None, "consensus_missing_matching_confidence"
    confidences = [confidence for confidence, _ in evidence]
    if min(confidences) < MINIMUM_CONSENSUS_CONFIDENCE:
        return None, "consensus_matching_confidence_below_threshold"
    confidence = min(confidences)
    return (
        _LabelChoice(
            expected=label,
            tier="consensus_silver",
            confidence=confidence,
            sample_weight=round(confidence, 6),
            provenance={
                "label_source": "materialized_cross_model_consensus",
                "policy_version": consensus.get("policy_version"),
                "valid_proposal_count": counts[0],
                "agreeing_model_count": counts[1],
                "independent_model_family_count": counts[2],
                "gold": False,
            },
        ),
        None,
    )


def _silver_choice(
    record: Mapping[str, Any], minimum_confidence: float
) -> tuple[_LabelChoice | None, str | None]:
    confidence = _confidence(record.get("confidence"))
    if confidence is None:
        return None, "silver_invalid_confidence"
    if confidence < minimum_confidence:
        return None, "silver_below_confidence_threshold"
    if "silver_label" not in record:
        return None, "silver_invalid_label"
    try:
        label = canonical_label(record.get("silver_label"))
    except PrivateDataError:
        return None, "silver_invalid_label"
    reasons = record.get("heuristic_reason_codes")
    if not isinstance(reasons, list) or not all(isinstance(value, str) for value in reasons):
        reasons = []
    return (
        _LabelChoice(
            expected=label,
            tier="grounded_silver",
            confidence=confidence,
            sample_weight=round(confidence * 0.75, 6),
            provenance={
                "label_source": "high_confidence_grounded_manifest_silver",
                "heuristic_reason_codes": list(reasons),
                "hard_negative_category": record.get("hard_negative_category"),
                "gold": False,
            },
        ),
        None,
    )


def _select_label(
    record: Mapping[str, Any], minimum_confidence: float
) -> tuple[_LabelChoice | None, tuple[str, ...]]:
    failures: list[str] = []
    factories = (
        lambda: _human_choice(record),
        lambda: _consensus_choice(record),
        lambda: _silver_choice(record, minimum_confidence),
    )
    sms = str(record["sms"])
    for factory in factories:
        choice, failure = factory()
        if failure is not None:
            failures.append(failure)
        if choice is None:
            continue
        grounding = grounding_errors(sms, choice.expected)
        if grounding:
            failures.extend(f"{choice.tier}:{reason}" for reason in grounding)
            continue
        return choice, tuple(failures)
    return None, tuple(failures)


def _category(choice: _LabelChoice, record: Mapping[str, Any]) -> str:
    if choice.expected is not None:
        return f"transaction:{choice.expected['type']}"
    hard_negative = record.get("hard_negative_category")
    suffix = hard_negative if isinstance(hard_negative, str) and hard_negative else "null"
    return f"null:{suffix}"


def _candidate_order(candidate: _Candidate) -> tuple[int, float, str]:
    return (
        TIER_PRIORITY[candidate.choice.tier],
        -candidate.choice.confidence,
        candidate.record_hash,
    )


def _cap_candidates(
    candidates: Sequence[_Candidate], config: BuildConfig
) -> tuple[list[_Candidate], Counter[str], dict[str, dict[str, int]]]:
    excluded: Counter[str] = Counter()
    before_categories = Counter(candidate.category for candidate in candidates)

    by_template: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_template[candidate.template_group].append(candidate)
    template_capped: list[_Candidate] = []
    for template_group in sorted(by_template):
        ordered = sorted(by_template[template_group], key=_candidate_order)
        template_capped.extend(ordered[: config.max_per_template])
        excluded["template_cap"] += max(0, len(ordered) - config.max_per_template)

    by_category: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in template_capped:
        by_category[candidate.category].append(candidate)
    category_capped: list[_Candidate] = []
    for category in sorted(by_category):
        ordered = sorted(by_category[category], key=_candidate_order)
        category_capped.extend(ordered[: config.max_per_category])
        excluded["category_cap"] += max(0, len(ordered) - config.max_per_category)

    transactions = [
        candidate for candidate in category_capped if candidate.choice.expected is not None
    ]
    nulls = [candidate for candidate in category_capped if candidate.choice.expected is None]
    if transactions:
        maximum_nulls = math.floor(len(transactions) * config.max_null_to_transaction_ratio)
        ordered_nulls = sorted(nulls, key=_candidate_order)
        excluded["null_balance_cap"] += max(0, len(ordered_nulls) - maximum_nulls)
        nulls = ordered_nulls[:maximum_nulls]
    balanced = sorted(transactions + nulls, key=lambda item: item.record_hash)
    after_categories = Counter(candidate.category for candidate in balanced)
    return (
        balanced,
        excluded,
        {
            "before_caps": dict(sorted(before_categories.items())),
            "after_caps": dict(sorted(after_categories.items())),
        },
    )


def _component_map(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        template_group = record.get("template_group")
        private_hashes = record.get("private_hashes")
        if (
            isinstance(template_group, str)
            and template_group
            and isinstance(private_hashes, Mapping)
            and isinstance(private_hashes.get("sender"), str)
            and private_hashes.get("sender")
        ):
            grouped[template_group].append(record)
    components = _sender_template_components(grouped)
    return {
        template_group: component
        for component in components
        for template_group in component
    }


def _stable_component_score(seed: int, component: tuple[str, ...]) -> str:
    value = f"{seed}\0" + "\0".join(component)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_candidates(
    candidates: Sequence[_Candidate],
    all_original_train_rows: Sequence[Mapping[str, Any]],
    config: BuildConfig,
) -> tuple[list[_Candidate], list[_Candidate], dict[str, Any]]:
    component_by_template = _component_map(all_original_train_rows)
    members: dict[tuple[str, ...], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        component = component_by_template.get(candidate.template_group)
        if component is None:
            raise PrivateSFTV2Error("a selected row is missing sender/template connectivity")
        members[component].append(candidate)
    if len(members) < 2:
        raise PrivateSFTV2Error(
            "at least two sender/template components are required for a sealed inner split"
        )

    ordered_components = sorted(
        members,
        key=lambda component: (_stable_component_score(config.seed, component), component),
    )
    target_dev_rows = max(1, min(len(candidates) - 1, round(len(candidates) * config.dev_fraction)))
    prefix_rows = 0
    cut_candidates: list[tuple[int, int]] = []
    for index, component in enumerate(ordered_components[:-1], start=1):
        prefix_rows += len(members[component])
        cut_candidates.append((abs(prefix_rows - target_dev_rows), index))
    _, cut = min(cut_candidates)
    dev_components = set(ordered_components[:cut])

    train: list[_Candidate] = []
    dev: list[_Candidate] = []
    for candidate in candidates:
        component = component_by_template[candidate.template_group]
        (dev if component in dev_components else train).append(candidate)
    train.sort(key=lambda item: item.record_hash)
    dev.sort(key=lambda item: item.record_hash)
    if not train or not dev:
        raise PrivateSFTV2Error("sender/template component split produced an empty partition")

    train_templates = {candidate.template_group for candidate in train}
    dev_templates = {candidate.template_group for candidate in dev}
    train_senders = {candidate.sender_hash for candidate in train}
    dev_senders = {candidate.sender_hash for candidate in dev}
    if train_templates & dev_templates or train_senders & dev_senders:
        raise PrivateSFTV2Error("sender/template component leakage detected after inner split")
    return train, dev, {
        "assignment_rule": "seeded_hash_ordered_sender_template_components",
        "component_count": len(members),
        "train_component_count": len(members) - len(dev_components),
        "dev_component_count": len(dev_components),
        "target_dev_rows": target_dev_rows,
        "actual_dev_rows": len(dev),
        "template_overlap_count": 0,
        "sender_overlap_count": 0,
    }


def _source_row(record: Mapping[str, Any], choice: _LabelChoice) -> dict[str, Any]:
    source_provenance = record.get("provenance")
    return {
        "sender": record["sender"],
        "sms": record["sms"],
        "expected": choice.expected,
        "sample_weight": choice.sample_weight,
        "confidence": choice.confidence,
        "label_tier": choice.tier,
        "source": {
            "dataset": "PRIVATE_DATA/lfm25/split_manifest.jsonl",
            "record_hash": record["record_hash"],
            "original_split": "train",
            "template_group": record["template_group"],
            "private_sender_hash": record["private_hashes"]["sender"],
        },
        "provenance": {
            "builder_version": BUILDER_VERSION,
            "label": dict(choice.provenance),
            "manifest": dict(source_provenance) if isinstance(source_provenance, Mapping) else {},
            "android_prefilter_accepted": True,
            "sample_weight_basis": "tier_and_label_confidence",
        },
    }


def _safe_prefilter_reason(result: Any) -> str:
    stage = getattr(result, "rejection_stage", None) or "unknown_stage"
    reason = getattr(result, "rejection_reason", None) or "unspecified"
    safe_stage = _SAFE_REASON_RE.sub("_", str(stage).casefold()).strip("_")
    safe_reason = _SAFE_REASON_RE.sub("_", str(reason).casefold()).strip("_")
    return f"android_prefilter:{safe_stage or 'unknown'}:{safe_reason or 'unspecified'}"


def _validate_train_record(record: Mapping[str, Any], seen_hashes: set[str]) -> str | None:
    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str) or not record_hash:
        return "train_missing_record_hash"
    if record_hash in seen_hashes:
        return "train_duplicate_record_hash"
    seen_hashes.add(record_hash)
    if not isinstance(record.get("sender"), str) or not isinstance(record.get("sms"), str):
        return "train_missing_sender_or_sms"
    if not record["sms"].strip():
        return "train_empty_sms"
    if not isinstance(record.get("template_group"), str) or not record["template_group"]:
        return "train_missing_template_group"
    private_hashes = record.get("private_hashes")
    if (
        not isinstance(private_hashes, Mapping)
        or not isinstance(private_hashes.get("sender"), str)
        or not private_hashes.get("sender")
    ):
        return "train_missing_private_sender_hash"
    return None


def _android_provenance(
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if supplied is not None:
        return dict(supplied)
    if _android_contract_provenance is None:
        raise PrivateSFTV2Error("the Android contract module is unavailable; refusing to drift")
    value = _android_contract_provenance()
    if not isinstance(value, Mapping):
        raise PrivateSFTV2Error("the Android contract provenance is invalid")
    return dict(value)


def build_private_sft_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    config: BuildConfig = BuildConfig(),
    prefilter: Callable[[str, str], Any] | None = None,
    android_provenance: Mapping[str, Any] | None = None,
) -> BuildResult:
    """Build private train/dev rows in memory without exposing row values."""

    config.validate()
    prefilter_fn = prefilter or _android_prefilter_sms
    if prefilter_fn is None:
        raise PrivateSFTV2Error("the Android prefilter is unavailable; refusing to drift")
    contract = _android_provenance(android_provenance)

    source_split_counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    label_rejections: Counter[str] = Counter()
    selected_tiers: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    original_train_rows: list[Mapping[str, Any]] = []
    original_test_hashes: set[str] = set()
    candidates: list[_Candidate] = []
    filter_pass_count = 0

    for record in records:
        split = record.get("split")
        split_name = str(split) if split in {"train", "dev", "test"} else "invalid"
        source_split_counts[split_name] += 1
        if split == "test":
            record_hash = record.get("record_hash")
            if isinstance(record_hash, str):
                original_test_hashes.add(record_hash)
            exclusions["original_test_sealed"] += 1
            continue
        if split == "dev":
            exclusions["original_dev_sealed"] += 1
            continue
        if split != "train":
            exclusions["invalid_original_split"] += 1
            continue

        error = _validate_train_record(record, seen_hashes)
        if error is not None:
            exclusions[error] += 1
            continue
        original_train_rows.append(record)
        result = prefilter_fn(str(record["sender"]), str(record["sms"]))
        accepted = getattr(result, "accepted", None)
        if not isinstance(accepted, bool):
            raise PrivateSFTV2Error("the Android prefilter returned an invalid result")
        if not accepted:
            exclusions[_safe_prefilter_reason(result)] += 1
            continue
        filter_pass_count += 1

        choice, failures = _select_label(record, config.minimum_silver_confidence)
        label_rejections.update(failures)
        if choice is None:
            exclusions["no_acceptable_grounded_label"] += 1
            continue
        selected_tiers[choice.tier] += 1
        candidates.append(
            _Candidate(record=record, choice=choice, category=_category(choice, record))
        )

    if not candidates:
        raise PrivateSFTV2Error("no original-train rows passed filtering and label safeguards")
    capped, cap_exclusions, category_counts = _cap_candidates(candidates, config)
    exclusions.update(cap_exclusions)
    if len(capped) < 2:
        raise PrivateSFTV2Error("selection caps left too few rows for a group-safe inner split")

    train_candidates, dev_candidates, split_report = _split_candidates(
        capped, original_train_rows, config
    )
    train_rows = [_source_row(candidate.record, candidate.choice) for candidate in train_candidates]
    dev_rows = [_source_row(candidate.record, candidate.choice) for candidate in dev_candidates]

    all_output_rows = train_rows + dev_rows
    materialized_hashes = {str(row["source"]["record_hash"]) for row in all_output_rows}
    sealed_overlap = materialized_hashes & original_test_hashes
    wrong_original_split = sum(
        row["source"].get("original_split") != "train" for row in all_output_rows
    )
    silver_dev_marked_gold = sum(
        row["label_tier"] != "human_gold"
        and row["provenance"]["label"].get("gold") is not False
        for row in dev_rows
    )
    if sealed_overlap or wrong_original_split or silver_dev_marked_gold:
        raise PrivateSFTV2Error("sealed-source or label-tier assertion failed")

    train_content = _jsonl_bytes(train_rows)
    dev_content = _jsonl_bytes(dev_rows)
    artifact_hashes = {
        "train": {
            "filename": OUTPUT_FILENAMES["train"],
            "rows": len(train_rows),
            "sha256": _sha256_bytes(train_content),
            "record_hash_set_sha256": _record_hash_set_sha256(train_rows),
        },
        "dev": {
            "filename": OUTPUT_FILENAMES["dev"],
            "rows": len(dev_rows),
            "sha256": _sha256_bytes(dev_content),
            "record_hash_set_sha256": _record_hash_set_sha256(dev_rows),
        },
    }
    output_tiers = Counter(str(row["label_tier"]) for row in all_output_rows)
    dev_tiers = Counter(str(row["label_tier"]) for row in dev_rows)
    split_label_kind_counts = {
        split: {
            "transaction": sum(row["expected"] is not None for row in rows),
            "null": sum(row["expected"] is None for row in rows),
        }
        for split, rows in (("train", train_rows), ("dev", dev_rows))
    }
    dev_null_rows = split_label_kind_counts["dev"]["null"]
    dev_evaluation = {
        "role": "silver_tuning_split_not_gold_benchmark",
        "contains_silver_labels": any(row["label_tier"] != "human_gold" for row in dev_rows),
        "gold_benchmark_claimed": False,
        "minimum_null_rows_for_ghost_diagnostic": 25,
        "underpowered_for_null_or_ghost_evaluation": dev_null_rows < 25,
        "warning": (
            "derived dev is for tuning only and is underpowered for null/ghost evaluation"
            if dev_null_rows < 25
            else None
        ),
    }
    sealed_test_assertion = {
        "passed": not sealed_overlap and wrong_original_split == 0,
        "assertion": (
            "original test rows were ineligible for prefiltering, label selection, "
            "sampling, and derived train/dev output"
        ),
        "original_test_rows": source_split_counts["test"],
        "original_test_rows_eligible": 0,
        "original_test_rows_materialized": len(sealed_overlap),
        "non_train_source_rows_materialized": wrong_original_split,
    }
    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "valid": True,
        "input_row_count": len(records),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "original_train_rows_valid_for_connectivity": len(original_train_rows),
        "android_filter_pass_rows": filter_pass_count,
        "labeled_candidates_before_caps": len(candidates),
        "materialized_row_count": len(all_output_rows),
        "split_row_counts": {"train": len(train_rows), "dev": len(dev_rows)},
        "label_tier_counts_before_caps": dict(sorted(selected_tiers.items())),
        "label_tier_counts": dict(sorted(output_tiers.items())),
        "dev_label_tier_counts": dict(sorted(dev_tiers.items())),
        "split_label_kind_counts": split_label_kind_counts,
        "dev_evaluation": dev_evaluation,
        "category_counts": category_counts,
        "exclusion_reasons": {key: value for key, value in sorted(exclusions.items()) if value},
        "label_candidate_rejections": {
            key: value for key, value in sorted(label_rejections.items()) if value
        },
        "inner_split": split_report,
        "sealed_test_assertion": sealed_test_assertion,
        "invariants": {
            "only_original_train_materialized": wrong_original_split == 0,
            "sealed_test_exclusion_passed": not sealed_overlap,
            "original_test_rows_materialized_count": len(sealed_overlap),
            "sender_overlap_between_inner_splits": False,
            "template_overlap_between_inner_splits": False,
            "android_prefilter_required": True,
            "transaction_labels_source_grounded": True,
            "silver_dev_rows_marked_gold": bool(silver_dev_marked_gold),
            "silver_dev_is_not_claimed_as_gold": silver_dev_marked_gold == 0,
            "raw_values_emitted_to_stdout": False,
            "outputs_private_local_only": True,
        },
    }
    metadata = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "dataset_state": "private_local_training_only",
        "release_authorized": False,
        "manifest_filename": DEFAULT_MANIFEST_NAME,
        "manifest_sha256": manifest_sha256,
        "builder_config": config.public_dict(),
        "builder_config_sha256": _config_sha256(config),
        "android_contract": contract,
        "artifacts": artifact_hashes,
        "label_tiers": {
            "human_gold": "approved human label with reviewer metadata",
            "consensus_silver": "accepted cross-model consensus; explicitly not gold",
            "grounded_silver": "high-confidence heuristic label; explicitly not gold",
        },
        "sample_weight_policy": {
            "human_gold": 1.0,
            "consensus_silver": "minimum matching proposal confidence",
            "grounded_silver": "0.75 * manifest heuristic confidence",
            "trainer_compatibility": (
                "top-level sample_weight is retained even if a trainer ignores extra fields"
            ),
        },
        "sealed_test_assertion": sealed_test_assertion,
    }
    return BuildResult(
        train_rows=train_rows,
        dev_rows=dev_rows,
        metadata=metadata,
        report=report,
    )


def _resolve_paths(
    repo_root: Path, manifest_path: Path, output_dir: Path
) -> tuple[Path, Path, Path]:
    repo_root = repo_root.resolve()
    private_root = (repo_root / "PRIVATE_DATA" / "lfm25").resolve()
    manifest = manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    output = output_dir if output_dir.is_absolute() else repo_root / output_dir
    manifest = ensure_within(manifest, private_root)
    output = ensure_within(output, private_root)
    if manifest.name != DEFAULT_MANIFEST_NAME or not manifest.is_file():
        raise PrivateSFTV2Error("input must be the local PRIVATE_DATA/lfm25 split manifest")
    if output == manifest or manifest in output.parents:
        raise PrivateSFTV2Error("private SFT output directory conflicts with the input manifest")
    require_private_ignore(repo_root, private_root)
    return repo_root, private_root, output


def run_private_sft_v2(
    *,
    repo_root: Path,
    manifest_path: Path,
    output_dir: Path,
    config: BuildConfig = BuildConfig(),
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Validate guardrails, build artifacts, and atomically persist unless dry-run."""

    _, _, resolved_output = _resolve_paths(repo_root, manifest_path, output_dir)
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path.is_absolute()
        else (repo_root.resolve() / manifest_path).resolve()
    )
    manifest_before = file_sha256(resolved_manifest)
    records = read_jsonl(resolved_manifest)
    if file_sha256(resolved_manifest) != manifest_before:
        raise PrivateSFTV2Error("the private manifest changed while it was being read")
    result = build_private_sft_v2(
        records,
        manifest_sha256=manifest_before,
        config=config,
    )
    if file_sha256(resolved_manifest) != manifest_before:
        raise PrivateSFTV2Error("the private manifest changed while artifacts were built")
    destinations = {
        name: resolved_output / filename for name, filename in OUTPUT_FILENAMES.items()
    }
    if dry_run:
        return {
            "dry_run": True,
            "wrote_files": False,
            "report": result.report,
            "metadata": result.metadata,
        }
    if not force and any(path.exists() for path in destinations.values()):
        raise PrivateSFTV2Error("private v2 SFT outputs already exist; pass --force to replace")

    resolved_output.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        resolved_output.chmod(0o700)
    except OSError:
        pass
    _atomic_write_jsonl(destinations["train"], result.train_rows)
    _atomic_write_jsonl(destinations["dev"], result.dev_rows)
    _atomic_write_json(destinations["metadata"], result.metadata)
    _atomic_write_json(destinations["report"], result.report)
    for path in destinations.values():
        try:
            path.chmod(0o600)
        except OSError:
            pass
    if file_sha256(destinations["train"]) != result.metadata["artifacts"]["train"]["sha256"]:
        raise PrivateSFTV2Error("written private train artifact failed its hash check")
    if file_sha256(destinations["dev"]) != result.metadata["artifacts"]["dev"]["sha256"]:
        raise PrivateSFTV2Error("written private dev artifact failed its hash check")
    return {
        "dry_run": False,
        "wrote_files": True,
        "report": result.report,
        "metadata": result.metadata,
    }


def safe_console_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate-only CLI output; never include source rows or messages."""

    report = value["report"]
    metadata = value["metadata"]
    return {
        "dry_run": bool(value["dry_run"]),
        "wrote_files": bool(value["wrote_files"]),
        "valid": bool(report["valid"]),
        "input_row_count": report["input_row_count"],
        "source_split_counts": report["source_split_counts"],
        "android_filter_pass_rows": report["android_filter_pass_rows"],
        "materialized_row_count": report["materialized_row_count"],
        "split_row_counts": report["split_row_counts"],
        "label_tier_counts": report["label_tier_counts"],
        "split_label_kind_counts": report["split_label_kind_counts"],
        "dev_evaluation": report["dev_evaluation"],
        "exclusion_reasons": report["exclusion_reasons"],
        "inner_split": report["inner_split"],
        "sealed_test_assertion": report["sealed_test_assertion"],
        "artifact_hashes": {
            split: details["sha256"]
            for split, details in metadata["artifacts"].items()
        },
    }
