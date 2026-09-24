"""SLM-primary extraction orchestration and persistence policy."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .account_resolution import (
    AccountCatalogEntry,
    DuplicateAssessment,
    DuplicateStatus,
    ExtractorAccountResolution,
    ResolutionStatus,
    assess_duplicate,
    resolve_account,
    transaction_fingerprint,
)
from .analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from .corpus.grouping import sender_family
from .extractor import (
    ExtractionResult,
    ExtractionValidationError,
    ExtractorFieldEvidence,
    NormalizedExtraction,
    SourceSpan,
    build_extractor_input,
    collect_grounded_extractor_fields,
    parse_and_normalize_extraction,
)
from .feedback import FieldRevisionV3
from .types import GateCheck, GateResult, TimestampProvenance


PROCESSING_RESULT_CONTRACT_V3 = "pocketfinancer.processing-result/3"
REVIEW_CASE_CONTRACT = "pocketfinancer.review-case/1"
REVIEW_CASE_CONTRACT_V2 = "pocketfinancer.review-case/2"


class RuntimeUnavailableError(RuntimeError):
    """The configured local model/runtime is not available."""


class RuntimeTruncatedError(RuntimeError):
    """The local runtime stopped at the configured output token limit."""


@dataclass(slots=True)
class CancellationToken:
    """Cooperative stop signal; late model output is always fenced."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class PersistenceContextV3:
    received_at_epoch_ms: int | None
    received_timestamp_provenance: TimestampProvenance
    account_resolution: ExtractorAccountResolution
    duplicate_assessment: DuplicateAssessment
    rollout_mode: str
    extractor_mode_valid: bool
    claim_ownership_current: bool
    configuration_hash_matches: bool
    operation_integrity_valid: bool = True


@dataclass(frozen=True, slots=True)
class PersistenceDecisionV3:
    result: GateResult
    primary_reason: str
    checks: tuple[GateCheck, ...]

    @property
    def safe_to_persist(self) -> bool:
        return self.result == GateResult.ELIGIBLE


@dataclass(frozen=True, slots=True)
class ReviewUserRevision:
    revision: int
    action: str
    actor_id_hash: str
    field_revisions: tuple[FieldRevisionV3, ...]
    created_at_epoch_ms: int
    previous_revision_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValueError("review user revision is invalid")
        if self.action not in {"confirm", "correct", "reject", "save_draft", "retry"}:
            raise ValueError("review user action is unsupported")
        if self.action == "correct" and not self.field_revisions:
            raise ValueError("review correction requires field revisions")
        if self.action not in {"correct", "save_draft"} and self.field_revisions:
            raise ValueError("review action cannot contain field revisions")
        if not _is_sha256(self.actor_id_hash) or (
            isinstance(self.created_at_epoch_ms, bool)
            or not isinstance(self.created_at_epoch_ms, int)
            or self.created_at_epoch_ms < 0
        ):
            raise ValueError("review user identity or timestamp is invalid")
        if self.revision == 1 and self.previous_revision_hash is not None:
            raise ValueError("first user revision cannot reference a predecessor")
        if self.revision > 1 and not _is_sha256(self.previous_revision_hash):
            raise ValueError("later user revision requires a predecessor hash")

    @property
    def revision_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "revision_hash": self.revision_hash}

    def _payload(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "action": self.action,
            "actor_id_hash": self.actor_id_hash,
            "field_revisions": [item.to_dict() for item in self.field_revisions],
            "created_at_epoch_ms": self.created_at_epoch_ms,
            "previous_revision_hash": self.previous_revision_hash,
        }


@dataclass(frozen=True, slots=True)
class ReviewCase:
    review_case_id: str
    operation_id_hash: str
    raw_sender: str
    source: str
    received_at_epoch_ms: int
    primary_reason: str
    reason_codes: tuple[str, ...]
    furthest_stage: str
    analyzer_suggestions: tuple[dict[str, Any], ...]
    extractor_suggestion: dict[str, Any] | None
    account_resolution: ExtractorAccountResolution | None
    received_timestamp_provenance: TimestampProvenance = TimestampProvenance.PLATFORM_RECEIVED
    revision: int = 1
    user_revisions: tuple[ReviewUserRevision, ...] = ()
    previous_revision_hash: str | None = None
    contract: str = REVIEW_CASE_CONTRACT

    def __post_init__(self) -> None:
        if not self.review_case_id or not _is_sha256(self.operation_id_hash) or not self.source:
            raise ValueError("review case identity or source is missing")
        if self.received_at_epoch_ms < 0 or self.revision < 1:
            raise ValueError("review case timestamp or revision is invalid")
        if not self.primary_reason or self.primary_reason not in self.reason_codes:
            raise ValueError("review case reasons are inconsistent")
        if self.revision == 1 and self.previous_revision_hash is not None:
            raise ValueError("first review revision cannot reference a predecessor")
        if self.revision > 1 and not _is_sha256(self.previous_revision_hash):
            raise ValueError("later review revision requires a predecessor hash")
        if self.revision != len(self.user_revisions) + 1:
            raise ValueError("review revision does not match its append-only user history")
        for index, user_revision in enumerate(self.user_revisions, start=1):
            if user_revision.revision != index:
                raise ValueError("review user revisions are not sequential")
            expected_previous = (
                None if index == 1 else self.user_revisions[index - 2].revision_hash
            )
            if user_revision.previous_revision_hash != expected_previous:
                raise ValueError("review user revision hash chain is invalid")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._payload_without_revision_hash()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_revision_hash(),
            "revision_hash": self.content_hash,
        }

    def _payload_without_revision_hash(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("received_timestamp_provenance")
        value["user_revisions"] = [item.to_dict() for item in self.user_revisions]
        value["receipt_timestamp"] = {
            "epoch_ms": value.pop("received_at_epoch_ms"),
            "provenance": self.received_timestamp_provenance.value,
            "read_only": True,
        }
        value["source_sha256"] = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        return value


@dataclass(frozen=True, slots=True)
class ReviewCaseV2(ReviewCase):
    """Successor Review representation with independently grounded fields."""

    field_evidence: tuple[ExtractorFieldEvidence, ...] = ()
    contract: str = REVIEW_CASE_CONTRACT_V2

    def __post_init__(self) -> None:
        ReviewCase.__post_init__(self)
        if self.contract != REVIEW_CASE_CONTRACT_V2:
            raise ValueError("successor review contract is unsupported")
        slm_fields = [item.field for item in self.field_evidence if item.origin == "slm"]
        if len(slm_fields) != len(set(slm_fields)):
            raise ValueError("successor review contains duplicate SLM field evidence")

    def _payload_without_revision_hash(self) -> dict[str, Any]:
        value = ReviewCase._payload_without_revision_hash(self)
        value["field_evidence"] = [item.to_dict() for item in self.field_evidence]
        return value


@dataclass(frozen=True, slots=True)
class ProcessingOutcome:
    status: str
    result: ExtractionResult | None
    persistence: PersistenceDecisionV3 | None
    review_case: ReviewCase | ReviewCaseV2 | None
    reason_codes: tuple[str, ...]
    model_invoked: bool
    received_at_epoch_ms: int
    received_timestamp_provenance: TimestampProvenance
    account_resolution: ExtractorAccountResolution | None = None
    duplicate_assessment: DuplicateAssessment | None = None


def evaluate_persistence_v3(
    result: ExtractionResult,
    context: PersistenceContextV3,
) -> PersistenceDecisionV3:
    """Evaluate extractor-era gates without analyzer/triage/selector dependencies."""

    transaction = result.transaction
    checks = (
        _check(
            "operation_integrity",
            context.operation_integrity_valid,
            "persistence_operation_integrity_invalid",
        ),
        _check(
            "configuration_hash",
            context.configuration_hash_matches,
            "persistence_configuration_hash_mismatch",
        ),
        _check(
            "claim_ownership",
            context.claim_ownership_current,
            "persistence_claim_ownership_invalid",
        ),
        _check(
            "extractor_mode",
            context.extractor_mode_valid,
            "persistence_extractor_mode_invalid",
        ),
        _check(
            "posted_extraction",
            result.decision == "posted" and transaction is not None,
            "persistence_not_posted",
        ),
        _check(
            "grounded_mandatory_fields",
            transaction is not None
            and all(
                span.text
                for span in (
                    transaction.amount_span,
                    transaction.direction_span,
                    transaction.account_span,
                )
            ),
            "persistence_grounded_fields_missing",
        ),
        _check(
            "valid_money",
            transaction is not None and transaction.minor_units > 0,
            "persistence_invalid_money",
        ),
        _check(
            "receipt_timestamp",
            context.received_at_epoch_ms is not None
            and context.received_at_epoch_ms >= 0
            and context.received_timestamp_provenance == TimestampProvenance.PLATFORM_RECEIVED,
            "persistence_receipt_timestamp_provenance_invalid",
        ),
        _check(
            "account_resolution",
            context.account_resolution.status == ResolutionStatus.UNIQUELY_RESOLVED
            and context.account_resolution.match_count == 1
            and context.account_resolution.account_id is not None,
            (
                "account_resolution_ambiguous"
                if context.account_resolution.status == ResolutionStatus.AMBIGUOUS
                else "account_resolution_unresolved"
            ),
        ),
        _check(
            "duplicate_assessment",
            context.duplicate_assessment.status == DuplicateStatus.CLEAR,
            (
                "duplicate_already_persisted"
                if context.duplicate_assessment.status == DuplicateStatus.ALREADY_PERSISTED
                else "duplicate_possible"
            ),
        ),
        _check(
            "rollout_mode",
            context.rollout_mode == "automatic",
            "persistence_blocked_by_rollout_mode",
        ),
    )
    failed = tuple(check for check in checks if not check.passed)
    primary = failed[0].reason_code if failed else "persistence_all_gates_passed"
    assert primary is not None
    integrity = {"operation_integrity", "configuration_hash", "claim_ownership"}
    if any(check.check in integrity for check in failed):
        gate_result = GateResult.INVALID_OPERATION
    elif result.decision == "none":
        gate_result = GateResult.NOT_POSTED
    elif any(check.check != "rollout_mode" for check in failed):
        gate_result = GateResult.REVIEW_REQUIRED
    elif failed:
        gate_result = GateResult.BLOCKED_BY_MODE
    else:
        gate_result = GateResult.ELIGIBLE
    return PersistenceDecisionV3(gate_result, primary, checks)


def processing_result_payload_v3(
    outcome: ProcessingOutcome,
) -> dict[str, Any]:
    """Serialize normalized semantics, immutable receipt time, and gate separately."""

    transaction = outcome.result.transaction if outcome.result is not None else None
    semantic = _semantic_payload(transaction)
    persistence = None
    if outcome.persistence is not None:
        persistence = {
            "result": outcome.persistence.result.value,
            "primary_reason": outcome.persistence.primary_reason,
            "checks": [asdict(check) for check in outcome.persistence.checks],
        }
    return {
        "contract": PROCESSING_RESULT_CONTRACT_V3,
        "status": outcome.status,
        "recognition_decision": outcome.result.decision if outcome.result else None,
        "semantic_result": semantic,
        "receipt_timestamp": {
            "epoch_ms": outcome.received_at_epoch_ms,
            "provenance": outcome.received_timestamp_provenance.value,
            "read_only": True,
        },
        "account_resolution": (
            asdict(outcome.account_resolution) if outcome.account_resolution else None
        ),
        "duplicate_assessment": (
            asdict(outcome.duplicate_assessment) if outcome.duplicate_assessment else None
        ),
        "automatic_persistence": persistence,
        "reason_codes": list(outcome.reason_codes),
    }


class ExtractionCoordinator:
    """One-pass coordinator for valid incoming SMS messages."""

    def __init__(
        self,
        analyzer: DeterministicSmsAnalyzer,
        model_invoke: Callable[[dict[str, Any], CancellationToken], str],
        *,
        primary_currency: str,
        enabled_profile_ids: tuple[str, ...],
        account_catalog: tuple[AccountCatalogEntry, ...] = (),
        rollout_mode: str = "shadow",
        review_contract: str = REVIEW_CASE_CONTRACT,
    ) -> None:
        if rollout_mode not in {"shadow", "review_only", "automatic"}:
            raise ValueError("rollout mode is unsupported")
        if analyzer.analysis_contract != ANALYSIS_CONTRACT_V2:
            raise ValueError("extractor coordinator requires sms-analysis/2")
        if review_contract not in {REVIEW_CASE_CONTRACT, REVIEW_CASE_CONTRACT_V2}:
            raise ValueError("review contract is unsupported")
        self.analyzer = analyzer
        self.model_invoke = model_invoke
        self.primary_currency = primary_currency
        self.enabled_profile_ids = enabled_profile_ids
        self.account_catalog = account_catalog
        self.rollout_mode = rollout_mode
        self.review_contract = review_contract

    def process(
        self,
        *,
        source: str,
        raw_sender: str,
        operation_id: str,
        operation_config_hash: str,
        received_at_epoch_ms: int,
        received_timestamp_provenance: TimestampProvenance,
        source_event_key: str,
        idempotency_key: str,
        is_outgoing: bool | None = False,
        input_valid: bool = True,
        cancellation: CancellationToken | None = None,
        persisted_idempotency_keys: frozenset[str] = frozenset(),
        persisted_source_event_keys: frozenset[str] = frozenset(),
        known_transaction_fingerprints: frozenset[str] = frozenset(),
        extractor_mode_valid: bool = True,
        claim_ownership_current: bool = True,
        configuration_hash_matches: bool = True,
    ) -> ProcessingOutcome:
        token = cancellation or CancellationToken()
        if (
            not input_valid
            or not isinstance(source, str)
            or not source
            or not isinstance(received_at_epoch_ms, int)
            or received_at_epoch_ms < 0
        ):
            return self._terminal(
                "bypassed",
                None,
                (("invalid_input",)),
                False,
                received_at_epoch_ms if isinstance(received_at_epoch_ms, int) else 0,
                received_timestamp_provenance,
            )
        if is_outgoing is True:
            return self._terminal(
                "bypassed",
                None,
                ("reliable_outgoing_metadata",),
                False,
                received_at_epoch_ms,
                received_timestamp_provenance,
            )

        analysis_kwargs: dict[str, Any] = {
            "operation_id": operation_id,
            "is_outgoing": False,
            "input_valid": True,
        }
        if self.analyzer.analysis_contract == ANALYSIS_CONTRACT_V2:
            analysis_kwargs.update(
                operation_config_hash=operation_config_hash,
                source_timestamp_epoch_ms=received_at_epoch_ms,
                source_timestamp_provenance=received_timestamp_provenance,
            )
        analysis = self.analyzer.analyze(source, **analysis_kwargs)
        request = build_extractor_input(
            source,
            analysis,
            sender_family=sender_family(raw_sender),
            primary_currency=self.primary_currency,
            enabled_profile_ids=self.enabled_profile_ids,
        )
        if token.cancelled:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "user_cancelled",
                "extractor_execution",
                False,
                status="interrupted",
            )
        try:
            raw_output = self.model_invoke(request, token)
        except RuntimeUnavailableError:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "runtime_unavailable",
                "extractor_execution",
                True,
            )
        except RuntimeTruncatedError:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "runtime_output_truncated",
                "extractor_execution",
                True,
            )
        except KeyboardInterrupt:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "process_interrupted",
                "extractor_execution",
                True,
                status="interrupted",
            )
        except Exception:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "runtime_failure",
                "extractor_execution",
                True,
            )
        if token.cancelled:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "user_cancelled",
                "extractor_execution",
                True,
                status="interrupted",
            )
        try:
            result = parse_and_normalize_extraction(
                raw_output,
                source,
                primary_currency=self.primary_currency,
                enabled_profile_ids=self.enabled_profile_ids,
            )
        except ExtractionValidationError as exc:
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                exc.reason_code,
                "extractor_validation",
                True,
                partial_fields=(
                    collect_grounded_extractor_fields(
                        raw_output,
                        source,
                        primary_currency=self.primary_currency,
                        enabled_profile_ids=self.enabled_profile_ids,
                    )
                    if self.review_contract == REVIEW_CASE_CONTRACT_V2
                    else ()
                ),
            )
        if result.decision == "none":
            empty_resolution = resolve_account(None, self.account_catalog)
            duplicate = _empty_duplicate(idempotency_key, source_event_key)
            context = PersistenceContextV3(
                received_at_epoch_ms,
                received_timestamp_provenance,
                empty_resolution,
                duplicate,
                self.rollout_mode,
                extractor_mode_valid,
                claim_ownership_current,
                configuration_hash_matches,
            )
            persistence = evaluate_persistence_v3(result, context)
            return ProcessingOutcome(
                "not_posted",
                result,
                persistence,
                None,
                ("extractor_none",),
                True,
                received_at_epoch_ms,
                received_timestamp_provenance,
                empty_resolution,
                duplicate,
            )
        if result.decision == "abstain":
            return self._review_failure(
                source,
                raw_sender,
                operation_id,
                received_at_epoch_ms,
                received_timestamp_provenance,
                analysis,
                "extractor_abstained",
                "extractor_validation",
                True,
                extractor_result=result,
            )

        assert result.transaction is not None
        resolution = resolve_account(result.transaction.account_reference, self.account_catalog)
        fingerprint = transaction_fingerprint(
            minor_units=result.transaction.minor_units,
            currency=result.transaction.currency,
            direction=result.transaction.direction.value,
            account_id=resolution.account_id,
            received_at_epoch_ms=received_at_epoch_ms,
        )
        duplicate = assess_duplicate(
            idempotency_key=idempotency_key,
            source_event_key=source_event_key,
            transaction_fingerprint_value=fingerprint,
            persisted_idempotency_keys=persisted_idempotency_keys,
            persisted_source_event_keys=persisted_source_event_keys,
            known_transaction_fingerprints=known_transaction_fingerprints,
        )
        context = PersistenceContextV3(
            received_at_epoch_ms,
            received_timestamp_provenance,
            resolution,
            duplicate,
            self.rollout_mode,
            extractor_mode_valid,
            claim_ownership_current,
            configuration_hash_matches,
        )
        persistence = evaluate_persistence_v3(result, context)
        reasons = tuple(
            check.reason_code for check in persistence.checks if check.reason_code is not None
        )
        if persistence.result == GateResult.ELIGIBLE:
            return ProcessingOutcome(
                "eligible",
                result,
                persistence,
                None,
                ("extractor_posted_valid",),
                True,
                received_at_epoch_ms,
                received_timestamp_provenance,
                resolution,
                duplicate,
            )
        review = _review_case(
            source,
            raw_sender,
            operation_id,
            received_at_epoch_ms,
            received_timestamp_provenance,
            analysis,
            reasons or (persistence.primary_reason,),
            "persistence_gate",
            result,
            resolution,
            review_contract=self.review_contract,
        )
        return ProcessingOutcome(
            "blocked" if persistence.result == GateResult.BLOCKED_BY_MODE else "review",
            result,
            persistence,
            review,
            review.reason_codes,
            True,
            received_at_epoch_ms,
            received_timestamp_provenance,
            resolution,
            duplicate,
        )

    @staticmethod
    def _terminal(
        status: str,
        result: ExtractionResult | None,
        reasons: tuple[str, ...],
        invoked: bool,
        received_at_epoch_ms: int,
        provenance: TimestampProvenance,
    ) -> ProcessingOutcome:
        return ProcessingOutcome(
            status,
            result,
            None,
            None,
            reasons,
            invoked,
            received_at_epoch_ms,
            provenance,
        )

    def _review_failure(
        self,
        source: str,
        raw_sender: str,
        operation_id: str,
        received_at_epoch_ms: int,
        provenance: TimestampProvenance,
        analysis: Any,
        reason: str,
        stage: str,
        invoked: bool,
        *,
        status: str = "review",
        extractor_result: ExtractionResult | None = None,
        partial_fields: tuple[ExtractorFieldEvidence, ...] = (),
    ) -> ProcessingOutcome:
        review = _review_case(
            source,
            raw_sender,
            operation_id,
            received_at_epoch_ms,
            provenance,
            analysis,
            (reason,),
            stage,
            extractor_result,
            None,
            review_contract=self.review_contract,
            partial_fields=partial_fields,
        )
        return ProcessingOutcome(
            status,
            extractor_result,
            None,
            review,
            (reason,),
            invoked,
            received_at_epoch_ms,
            provenance,
        )


class SuccessorExtractionCoordinator(ExtractionCoordinator):
    """Automatic successor policy using the existing routing engine."""

    def __init__(
        self,
        analyzer: DeterministicSmsAnalyzer,
        model_invoke: Callable[[dict[str, Any], CancellationToken], str],
        *,
        primary_currency: str,
        enabled_profile_ids: tuple[str, ...],
        account_catalog: tuple[AccountCatalogEntry, ...] = (),
    ) -> None:
        super().__init__(
            analyzer,
            model_invoke,
            primary_currency=primary_currency,
            enabled_profile_ids=enabled_profile_ids,
            account_catalog=account_catalog,
            rollout_mode="automatic",
            review_contract=REVIEW_CASE_CONTRACT_V2,
        )


def _review_case(
    source: str,
    raw_sender: str,
    operation_id: str,
    received_at_epoch_ms: int,
    received_timestamp_provenance: TimestampProvenance,
    analysis: Any,
    reasons: tuple[str, ...],
    stage: str,
    result: ExtractionResult | None,
    resolution: ExtractorAccountResolution | None,
    *,
    review_contract: str = REVIEW_CASE_CONTRACT,
    partial_fields: tuple[ExtractorFieldEvidence, ...] = (),
) -> ReviewCase | ReviewCaseV2:
    review_id = hashlib.sha256(
        f"{operation_id}\0{hashlib.sha256(source.encode()).hexdigest()}".encode()
    ).hexdigest()[:24]
    candidate_suggestions = tuple(
        {
            "kind": candidate.kind.value,
            "analyzer_version": analysis.contract,
            "span": _source_span_payload(
                SourceSpan(
                    candidate.evidence.start_char,
                    candidate.evidence.end_char,
                    candidate.evidence.text,
                )
            )
            if candidate.evidence
            else None,
            "clause": candidate.clause_id,
            "suggested_interpretation": candidate.value,
            "provenance": {
                "candidate_id": candidate.candidate_id,
                "analyzer_kind": "candidate",
            },
        }
        for candidate in analysis.candidates
    )
    cue_suggestions = tuple(
        {
            "kind": cue.kind,
            "analyzer_version": analysis.contract,
            "span": _source_span_payload(
                SourceSpan(
                    cue.evidence.start_char,
                    cue.evidence.end_char,
                    cue.evidence.text,
                )
            ),
            "clause": cue.clause_id,
            "suggested_interpretation": {"reason_code": cue.reason_code},
            "provenance": {
                "cue_id": cue.cue_id,
                "analyzer_kind": "cue",
            },
        }
        for cue in analysis.cues
    )
    analyzer_suggestions = candidate_suggestions + cue_suggestions
    values: dict[str, Any] = {
        "review_case_id": review_id,
        "operation_id_hash": hashlib.sha256(operation_id.encode()).hexdigest(),
        "raw_sender": raw_sender,
        "source": source,
        "received_at_epoch_ms": received_at_epoch_ms,
        "primary_reason": reasons[0],
        "reason_codes": reasons,
        "furthest_stage": stage,
        "analyzer_suggestions": analyzer_suggestions,
        "extractor_suggestion": _semantic_payload(result.transaction if result else None),
        "account_resolution": resolution,
        "received_timestamp_provenance": received_timestamp_provenance,
    }
    if review_contract == REVIEW_CASE_CONTRACT:
        return ReviewCase(**values)
    if review_contract != REVIEW_CASE_CONTRACT_V2:
        raise ValueError("review contract is unsupported")
    slm_fields = (
        partial_fields
        if partial_fields
        else _normalized_field_evidence(result.transaction if result else None)
    )
    analyzer_fields = tuple(
        ExtractorFieldEvidence(
            field=candidate.kind.value,
            source_span=SourceSpan(
                candidate.evidence.start_char,
                candidate.evidence.end_char,
                candidate.evidence.text,
            ),
            normalized_value=candidate.value,
            validation_state="suggestion",
            originating_stage="analysis_advisory",
            origin="advisory_analyzer",
        )
        for candidate in analysis.candidates
        if candidate.kind.value in {"amount", "direction", "account", "counterparty"}
        and candidate.evidence is not None
    )
    return ReviewCaseV2(**values, field_evidence=slm_fields + analyzer_fields)


def _semantic_payload(transaction: NormalizedExtraction | None) -> dict[str, Any] | None:
    if transaction is None:
        return None
    return {
        "money": {
            "minor_units": transaction.minor_units,
            "currency": transaction.currency,
        },
        "direction": transaction.direction.value,
        "account_reference": transaction.account_reference,
        "counterparty": transaction.counterparty,
        "evidence": {
            "amount": _source_span_payload(transaction.amount_span),
            "direction": _source_span_payload(transaction.direction_span),
            "account": _source_span_payload(transaction.account_span),
            "counterparty": _source_span_payload(transaction.counterparty_span),
        },
    }


def _normalized_field_evidence(
    transaction: NormalizedExtraction | None,
) -> tuple[ExtractorFieldEvidence, ...]:
    if transaction is None:
        return ()
    fields = [
        ExtractorFieldEvidence(
            "amount",
            transaction.amount_span,
            {"minor_units": transaction.minor_units, "currency": transaction.currency},
            "valid",
            "normalization",
        ),
        ExtractorFieldEvidence(
            "direction",
            transaction.direction_span,
            transaction.direction.value,
            "valid",
            "normalization",
        ),
        ExtractorFieldEvidence(
            "account",
            transaction.account_span,
            transaction.account_reference,
            "valid",
            "normalization",
        ),
    ]
    if transaction.counterparty_span is not None:
        fields.append(
            ExtractorFieldEvidence(
                "counterparty",
                transaction.counterparty_span,
                transaction.counterparty,
                "valid",
                "normalization",
            )
        )
    return tuple(fields)


def _source_span_payload(span: SourceSpan | None) -> dict[str, Any] | None:
    return span.to_dict() if span is not None else None


def _empty_duplicate(idempotency_key: str, source_event_key: str) -> DuplicateAssessment:
    return DuplicateAssessment(
        DuplicateStatus.CLEAR,
        idempotency_key,
        source_event_key,
        hashlib.sha256(b"not-posted").hexdigest(),
    )


def _check(name: str, passed: bool, reason: str) -> GateCheck:
    return GateCheck(name, passed, None if passed else reason)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _is_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
