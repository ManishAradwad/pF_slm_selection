# Documentation map

The repository contains the active SMS-processing foundation plus historical
model-selection/fine-tuning evidence. The root `README.md` is the human onboarding page, `AGENTS.md` contains
tool-neutral repository policy, and this page indexes durable technical evidence.
No agent-specific file is a separate source of truth.

Development workflow: [repository instructions](../AGENTS.md) and
[contribution guide](../CONTRIBUTING.md).

## Read in this order

1. [SMS Processing Architecture](architecture/SMS_PROCESSING_ARCHITECTURE.md) - the
   active end-to-end host/analyzer/model/review path.
2. [Grounded Candidate Selector Contract](contracts/GROUNDED_CANDIDATE_SELECTOR_CONTRACT.md) -
   the compact one-pass output and strict reconstruction rules.
3. [Currency Context and Provenance](architecture/CURRENCY_CONTEXT_AND_PROVENANCE.md) -
   explicit user currency snapshots and override behavior.
4. [Data Taxonomy and Canonical Labels](architecture/DATA_TAXONOMY_AND_CANONICAL_LABELS.md) -
   weak facets, human truth, and target projection.
5. [Workbench Requirements and Data Flow](architecture/WORKBENCH_REQUIREMENTS_AND_DATA_FLOW.md) -
   the local corpus browser, blind review, revision, backup, and export design.
6. [SMS Processing Execution Plan](plans/SMS_PROCESSING_EXECUTION_PLAN.md) - the
   single active plan, gates, post-workbench phases, and next-session prompt.
7. [SMS Processing Decision Log](architecture/SMS_PROCESSING_DECISION_LOG.md) -
   evidence, rejected alternatives, risks, and native responsibilities.
8. [Unified command map](../scripts/README.md) - active SMS/workbench commands and
   historical model-research commands.
9. [Historical evidence index](history/SMS_PROCESSING_EVIDENCE_INDEX.md) - how to
   interpret measured Phase/Candidate/Semantic artifacts without treating them as active.
10. [Annotation Handbook V1](guides/ANNOTATION_HANDBOOK_V1.md) - the historical decision,
   span, uncertainty, QC, and adjudication rules human reviewers must follow.
11. [Local annotation workbench](guides/LOCAL_ANNOTATION_WORKBENCH.md) - the historical
   strictly local blinded-test and training-curation operating procedure, including
   served-URL troubleshooting, an invented-data UI smoke launcher, opt-in
   source-assisted prefill, recovery, delayed QC, export, and final-import gates.
12. [Experiment catalog](experiments/EXPERIMENT_CATALOG.md) - historical dataset lineage,
   trusted results, invalidated runs, and what each count means.
13. [Candidate Protocol V1 controlled run](experiments/POCKETFINANCER_LFM25_350M_CANDIDATE_PROTOCOL_V1.md) -
   the executed 2026-08-08 three-seed comparison, negative safety-gate result,
   host packaging evidence, and remaining mobile-runtime and human-gold gates.
14. [LFM2.5-2.6B Base LoRA diagnostic](experiments/POCKETFINANCER_LFM25_2_6B_R16_S17.md) -
   the executed 2026-08-05 controlled run, untouched-Base comparison, memory
   probe, and remaining human-gold, device, and deployment gates.
15. [Android-aligned 350M LoRA run](experiments/POCKETFINANCER_A9_LORA_R16_S17.md) -
   the completed RTX 4070 training run, app-interpreted scores, GGUF comparisons,
   BOS/grammar ablations, and next decision gates.
16. [LFM2.5-350M pipeline report](LFM25_350M_PIPELINE_V2.md) - detailed chronology,
   methods, metrics, and artifacts. Read its Android-parity correction first.
17. [Fine-tuning primer](guides/FINE_TUNING_PRIMER.md) - SFT, LoRA, QLoRA, loss,
   train/dev/test, and why data quality matters more than row count alone.
18. [Android runtime audit](architecture/ANDROID_RUNTIME_AUDIT.md) - historical
   `a6c8a11` findings plus the current-profile correction.
19. [Repository layout](architecture/REPOSITORY_LAYOUT.md) - current source/private boundaries.
20. [LFM2.5-2.6B evaluation plan](experiments/LFM25_2_6B_EVALUATION_PLAN.md) - the
    pre-run design retained for provenance; use the executed report above for
    results and current status.
21. [Historical general GGUF sweep](history/GENERAL_GGUF_BENCHMARK_2026-04-25.md) -
    dated pre-app-contract model-slate evidence retained for audit only.

## Status language

- **Current** means the data and implementation passed the latest local grounding
  and provenance checks. It does not mean production-ready.
- **Historical** means useful evidence under an older contract.
- **Invalidated** means a discovered data or contract flaw prevents using the run
  for a model-quality conclusion.
- **Diagnostic only** means the experiment was intentionally useful for debugging
  but is unsafe to ship or compare as a final system.

The 203-row regression set has been repeatedly consulted. It is locked for
compatibility checking, not a fresh gold test set.

## Privacy

Reports and checked-in docs use aggregate counts only. Raw SMS, identifiers,
per-row private predictions, private manifests, adapters, and model checkpoints
remain local and ignored. Nothing in this repository authorizes publication.
