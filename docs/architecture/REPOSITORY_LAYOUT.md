# Repository layout and migration plan

## Current safe boundaries

The active checkout is native WSL2/ext4. Keep these local artifact roots in place:

```text
PRIVATE_DATA/       raw-derived private manifests and local datasets
MODELS/             downloaded HF/GGUF weights
TRAINING_ARTIFACTS/ adapters, merged checkpoints, training manifests
RESULTS/            per-run metrics and private samples
PUBLIC_CANDIDATE/   unreleased candidate public-data work
UPSTREAM/           pinned external source trees such as llama.cpp
```

They are ignored and protected by the repository safety checks. Moving private
inputs requires a separate hash-verified, recoverable migration.

The checked-in implementation currently has three tracks:

| Track | Main entry point | Status |
|---|---|---|
| General GGUF model slate | `scripts/evaluate.sh` | Useful historical benchmark |
| Short-prompt LFM training | `scripts/run_lfm25_experiments.py` | Legacy only |
| Prompt-aligned direct/candidate SFT | contract-specific builders, trainer, evaluators | Active research |

## Why no mass move in this pass

Most v2 work is still untracked, several historical files are modified, tests import
symbols from scripts, and provenance hashes bind exact source paths. A mass rename
would mix behavior changes with mechanical movement and make old artifacts harder
to audit. This pass adds navigation and status metadata while keeping every command
working.

## Target package

The model-specific `lfm25` package should eventually become a model-agnostic
`src/pf_slm` package. Model size then belongs in configuration, not duplicated code.

```text
src/pf_slm/
  contracts/      schema, PocketFinancer runtime, candidate protocol, legacy prompt
  data/           private manifests, labeling, SFT builders, synthetic curriculum
  training/       datasets, completion loss, LoRA/QLoRA, merge
  evaluation/     metrics, HF proxy, GGUF runtime, comparisons
  runtime/        llama.cpp integration and conversion
  provenance.py
  cli.py

configs/
  models/         immutable model/revision records
  contracts/      versioned app/runtime settings and hashes
  experiments/    dataset + model + training + evaluation declarations

tasks/financial_sms/
  task.yaml
  grammar.gbnf
  regression.jsonl
```

`lfm25` and current script paths should remain compatibility shims for at least one
migration cycle. Historical artifacts stay readable; new runs receive new source
hashes and semantic run IDs.

## Run identity

Future run IDs should state the variables that matter and avoid mutable words such
as `clean`, `final`, and unexplained version numbers. For example:

```text
lfm2-350m__candidate-selector__r16-lr1e4-s17__data-c48ea5d5
```

Every run manifest should include model revision, contract hash, dataset hashes,
split policy, label provenance, training/loss config, seed, selected checkpoint,
decode settings, evaluator hash, and status.

## Migration sequence

1. Verify and snapshot the current v2 implementation.
2. Maintain the README, documentation map, command map, and experiment catalog.
3. Introduce `src/pf_slm` and move reusable internals behind `lfm25` shims.
4. Move CLI implementations behind compatibility wrappers.
5. Split tests into unit, integration, runtime, and safety layers.
6. Normalize only future artifact interiors and run IDs.
7. Move private root inputs only with explicit approval and post-copy hashes.
