# Repository layout

## Current safe boundaries

CUDA/model work uses the native WSL2/ext4 checkout. Pure analysis, corpus, and
workbench work may run in the local macOS checkout containing the private archive.
Keep these local artifact roots in place:

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

The checked-in implementation has one active SMS foundation and historical model tracks:

| Track | Main entry point | Status |
|---|---|---|
| Shared SMS processing and review | `scripts/run_sms_processing.py` | **Active** |
| General GGUF model slate | `scripts/evaluate.sh` | Historical benchmark |
| Short-prompt LFM training | `scripts/run_lfm25_experiments.py` | Legacy |
| Prompt-aligned direct/candidate SFT | contract-specific builders, trainer, evaluators | Historical research |

## Active source package

```text
src/pocketfinancer_sms/
  analyzer.py       source-preserving shared analysis
  currency.py       exact money and explicit currency context
  triage.py         invoke/discard/retain-review policy
  selector.py       compact ID output and reconstruction
  persistence.py    automatic-save safety gate
  labels.py         weak taxonomy and canonical human truth
  trace.py          immutable observable processing stages
  feedback.py       append-only future native feedback
  corpus/           manifest, grouping, pools, and reports
  workbench/        SQLite service and local browser UI
  provenance.py
  cli.py

configs/sms_processing/
  contracts/        executable JSON schemas
  currency/         supported ISO-4217 subset
  profiles/         core and locale extensions
  archive-india-inr.json
```

`lfm25` remains in place so measured experiments and reports are reproducible. New
SMS behavior must not be added there. Historical artifacts stay readable and carry
supersession context through the evidence index.

## Run identity

Future run IDs should state the variables that matter and avoid mutable words such
as `clean`, `final`, and unexplained version numbers. For example:

```text
lfm2-350m__candidate-selector__r16-lr1e4-s17__data-c48ea5d5
```

Every run manifest should include model revision, contract hash, dataset hashes,
split policy, label provenance, training/loss config, seed, selected checkpoint,
decode settings, evaluator hash, and status.

## Cleanup boundary

Remove obsolete active plans, invalid generated private datasets, overlapping
splits, and superseded exploratory builders after canonical-copy and hash checks.
Do not remove measured reports or the code/config needed to interpret them. Do not
move the raw archive or reviewed annotations without an explicit, recoverable,
hash-verified migration.
