# Documentation map

The repository contains historical model-selection work and a newer fine-tuning
pipeline. This page is the human-facing index; `CLAUDE.md` remains operational
context for coding agents and is not the experiment catalog.

## Read in this order

1. [Model-improvement roadmap](architecture/MODEL_IMPROVEMENT_ROADMAP.md) - the
   product objective, today's direct fine-tuning path, and later architecture work.
2. [Unified command map](../scripts/README.md) - the one app-first pipeline and its
   build, train, evaluation, merge, conversion, and GGUF stages.
3. [Experiment catalog](experiments/EXPERIMENT_CATALOG.md) - dataset lineage,
   trusted results, invalidated runs, and what each count means.
4. [Latest Android-aligned LoRA run](experiments/POCKETFINANCER_A9_LORA_R16_S17.md) -
   the completed RTX 4070 training run, app-interpreted scores, GGUF comparisons,
   BOS/grammar ablations, and next decision gates.
5. [LFM2.5-350M pipeline report](LFM25_350M_PIPELINE_V2.md) - detailed chronology,
   methods, metrics, and artifacts. Read its Android-parity correction first.
6. [Fine-tuning primer](guides/FINE_TUNING_PRIMER.md) - SFT, LoRA, QLoRA, loss,
   train/dev/test, and why data quality matters more than row count alone.
7. [Android runtime audit](architecture/ANDROID_RUNTIME_AUDIT.md) - historical
   `a6c8a11` findings plus the current-profile correction.
8. [Repository layout](architecture/REPOSITORY_LAYOUT.md) - current boundaries and
   the staged move toward a model-agnostic package.
9. [LFM2.5-2.6B evaluation plan](experiments/LFM25_2_6B_EVALUATION_PLAN.md) - a
   future, like-for-like test of the new agentic model and its Base checkpoint.

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
