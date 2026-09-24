"""Production-intended, local-first SMS processing foundation."""

from .analyzer import DeterministicSmsAnalyzer
from .currency import CurrencyContext
from .extractor import (
    ExtractorFieldEvidence,
    SourceSpan,
    build_extractor_input,
    collect_grounded_extractor_fields,
    parse_and_normalize_extraction,
)
from .persistence import evaluate_persistence, processing_result_payload
from .processing_v3 import (
    ExtractionCoordinator,
    SuccessorExtractionCoordinator,
    evaluate_persistence_v3,
    processing_result_payload_v3,
)
from .selector import parse_and_reconstruct
from .triage import evaluate_triage

__all__ = [
    "CurrencyContext",
    "DeterministicSmsAnalyzer",
    "ExtractionCoordinator",
    "ExtractorFieldEvidence",
    "SourceSpan",
    "SuccessorExtractionCoordinator",
    "build_extractor_input",
    "collect_grounded_extractor_fields",
    "evaluate_persistence",
    "evaluate_persistence_v3",
    "evaluate_triage",
    "parse_and_normalize_extraction",
    "parse_and_reconstruct",
    "processing_result_payload",
    "processing_result_payload_v3",
]
