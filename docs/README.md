# Documentation map

The repository contains historical model-selection work and a newer fine-tuning
pipeline. The root `README.md` is the human onboarding page, `AGENTS.md` contains
tool-neutral repository policy, and this page indexes durable technical evidence.
No agent-specific file is a separate source of truth.

Development workflow: [repository instructions](../AGENTS.md) and
[contribution guide](../CONTRIBUTING.md).

## Read in this order

1. [Model-improvement roadmap](architecture/MODEL_IMPROVEMENT_ROADMAP.md) - the
   product objective, today's direct fine-tuning path, and later architecture work.
2. [Unified command map](../scripts/README.md) - the one app-first pipeline and its
   build, train, evaluation, merge, conversion, and GGUF stages.
3. [Annotation Handbook V1](guides/ANNOTATION_HANDBOOK_V1.md) - the decision,
   span, uncertainty, QC, and adjudication rules human reviewers must follow.
4. [Local annotation workbench](guides/LOCAL_ANNOTATION_WORKBENCH.md) - the
   strictly local blinded-test and training-curation operating procedure, including
   served-URL troubleshooting, an invented-data UI smoke launcher, opt-in
   source-assisted prefill, recovery, delayed QC, export, and final-import gates.
5. [Experiment catalog](experiments/EXPERIMENT_CATALOG.md) - dataset lineage,
   trusted results, invalidated runs, and what each count means.
6. [Candidate Protocol V1 controlled run](experiments/POCKETFINANCER_LFM25_350M_CANDIDATE_PROTOCOL_V1.md) -
   the executed 2026-08-08 three-seed comparison, negative safety-gate result,
   host packaging evidence, and remaining mobile-runtime and human-gold gates.
7. [LFM2.5-2.6B Base LoRA diagnostic](experiments/POCKETFINANCER_LFM25_2_6B_R16_S17.md) -
   the executed 2026-08-05 controlled run, untouched-Base comparison, memory
   probe, and remaining human-gold, device, and deployment gates.
8. [Android-aligned 350M LoRA run](experiments/POCKETFINANCER_A9_LORA_R16_S17.md) -
   the completed RTX 4070 training run, app-interpreted scores, GGUF comparisons,
   BOS/grammar ablations, and next decision gates.
9. [LFM2.5-350M pipeline report](LFM25_350M_PIPELINE_V2.md) - detailed chronology,
   methods, metrics, and artifacts. Read its Android-parity correction first.
10. [Fine-tuning primer](guides/FINE_TUNING_PRIMER.md) - SFT, LoRA, QLoRA, loss,
   train/dev/test, and why data quality matters more than row count alone.
11. [Android runtime audit](architecture/ANDROID_RUNTIME_AUDIT.md) - historical
   `a6c8a11` findings plus the current-profile correction.
12. [Repository layout](architecture/REPOSITORY_LAYOUT.md) - current boundaries and
   the staged move toward a model-agnostic package.
13. [LFM2.5-2.6B evaluation plan](experiments/LFM25_2_6B_EVALUATION_PLAN.md) - the
    pre-run design retained for provenance; use the executed report above for
    results and current status.
14. [Historical general GGUF sweep](history/GENERAL_GGUF_BENCHMARK_2026-04-25.md) -
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
