# Repository Development Instructions

This is the canonical repository guidance for humans and automated development
tools. It is intentionally independent of any editor, coding agent, or model
provider. Read this file before changing code, data builders, evaluation behavior,
or experiment documentation.

## Codex collaboration model profile

This repository currently uses the following Codex collaboration profile
(recorded 2026-08-31). It governs the coding agents working on this repository,
not the models evaluated, trained, or deployed by this project.

- The primary/root agent uses `gpt-5.6-sol` with `medium` reasoning. It owns
  scope, integration, and final verification.
- An `explorer` subagent uses `gpt-5.6-luna` with `medium` reasoning for
  read-only repository discovery and behavior tracing.
- A `worker` subagent uses `gpt-5.6-terra` with `medium` reasoning for bounded
  implementation and verification tasks.
- A `reviewer` subagent uses `gpt-5.6-terra` with `high` reasoning for read-only
  correctness, security, regression, and test review.
- A `default` subagent inherits the parent agent's model and reasoning effort
  unless the spawning task explicitly overrides them.

The active Codex runtime and task instructions are authoritative if this dated
profile differs from the live configuration. A model or reasoning override does
not change repository rules, task scope, permissions, or review requirements;
record any deliberate override in the task handoff.

## Authority and source of truth

The active production-intended SMS architecture is the shared foundation in
`src/pocketfinancer_sms`, its schemas under `configs/sms_processing`, and the
descriptive documents indexed by
`docs/architecture/SMS_PROCESSING_ARCHITECTURE.md`. Android and iOS have not yet
integrated it. The locked Android profile at
`configs/contracts/pocketfinancer-android-current.json` remains the authority for
reproducing the currently deployed/historical native path and measured experiments;
it is not the intended future shared contract.

Use this precedence when sources disagree:

1. Executable shared SMS code, schemas, tests, and corpus-run configuration here.
2. The active SMS architecture, selector contract, taxonomy, workbench, execution
   plan, and decision log.
3. For legacy/native parity only, committed app source at the profile's pinned
   revision and `lfm25/android_contract.py`.
4. The experiment catalog and dated historical run reports.

Verify prose against current code before relying on it. Do not silently update the
profile to match a different Android checkout; audit the app change and version the
contract deliberately.

## Project boundary

The active SMS foundation is operated with `scripts/run_sms_processing.py`. It
builds the ignored canonical corpus, initializes/serves the local workbench, and
creates private backups/exports. It does not train or deploy a model.

The older app-facing research workflow declared in
`configs/pipelines/pocketfinancer-lfm2.5-350m.json` and orchestrated by
`scripts/run_pocketfinancer_pipeline.py` remains reproducible historical model
research. Its stages must not be presented as the active data or product architecture.

The broad GGUF slate and older short-prompt experiments remain historical tools.
They must not be presented as current Android-runtime evidence. Keep reusable
logic model-agnostic: model identity, thinking behavior, quantization, and limits
belong in configuration or versioned profiles rather than hidden conditionals.

`lm-evaluation-harness/` and `UPSTREAM/` are external source trees. Prefer
out-of-tree extensions and do not edit vendored/upstream code unless the task
specifically requires it and the corresponding lock/provenance record is updated.

## Working environment

CUDA training, conversion, and legacy model evaluation use the native WSL2 repository:

```text
/home/tojinotzenin/pF_slm_selection
```

Activate that environment with `source scripts/activate_wsl.sh`. Pure-Python SMS
analysis, corpus construction, the local workbench, documentation, and lightweight
tests may run from the local macOS checkout containing the private archive. Never
copy private data between checkouts merely to satisfy an environment convention.

There are three verification tiers:

1. Lightweight CI: safety, Ruff, ShellCheck, and unit tests without Torch/model
   downloads.
2. Local ML: CUDA/backward probes, data materialization, training, and HF checks.
3. Runtime/device: GGUF parity checks and Android phone measurements.

Do not turn a higher tier into a prerequisite for documentation or pure unit work.
Never describe HF timing as GGUF timing or desktop host timing as phone timing.

## End-to-end change discipline

Trace every material change across its full path: Android source/profile, prompt
serialization, prefilter, dataset construction, labels and split policy, training
objective, checkpoint selection, merge, conversion, quantization, evaluator,
parser, metrics, reports, and deployment gate.

For long-running or resumable work, define behavior for fresh runs, existing
outputs, interruption, retry, partial artifacts, and provenance mismatch. Prefer
stage-by-stage execution. Never overwrite a nonempty training run implicitly or
reuse cached results whose hashes/configuration no longer match.

Keep experiment comparisons controlled. Record model revision, contract and data
hashes, seed, loss/training settings, selected checkpoint, decode settings,
quantization, evaluator hash, and runtime-parity limitations. Change one class of
variable at a time or label the comparison non-causal.

Add tests at the lowest practical layer, then run the relevant integration guard.
Update the experiment catalog when a result's trust status changes. Preserve
historical reports; add a clear supersession banner instead of rewriting old runs
as though they used a newer contract.

## Data, privacy, and publication

The following roots contain local or generated material and must remain ignored:

```text
PRIVATE_DATA/       raw-derived datasets, review queues, split manifests
PUBLIC_CANDIDATE/   unreleased dataset candidates
MODELS/             downloaded model/tokenizer/GGUF assets
TRAINING_ARTIFACTS/ adapters, checkpoints, merged weights, conversion outputs
RESULTS/            aggregate and per-row predictions/benchmarks
UPSTREAM/           locally pinned external source trees
```

Do not print or commit raw SMS, senders, identifiers, per-row private predictions,
stable source mappings, credentials, or model weights. Do not send private data to
hosted APIs, telemetry, W&B, remote labeling, or cloud inference. Synthetic test
fixtures must not be copied from private rows.

`DATA/extraction_ds.jsonl` is a grandfathered, repeatedly consulted 203-row
regression fixture. Never train on it or call it a fresh test. Production claims
require a newly adjudicated, template/sender-held-out human-gold test.

`PRIVATE_DATA/sms_processing` contains the canonical manifest, deterministic
analysis, pool views, reports, SQLite workbench, backups, and exports. Every row
remains local. Queue files are views over the canonical manifest and final SFT
targets are forbidden until submitted labels project without fallback.

Dataset preparation and local model work do not authorize upload, publication,
release, or deployment. Those actions require an explicit user decision after
privacy, data-rights, model-license, and target-device review.

## Required checks

From the activated WSL environment, the normal code/documentation gate is:

```bash
python scripts/check_repo_safety.py
ruff check .
ruff check --select E4,E7,E9,F lfm25 src scripts tests
pytest -q
git diff --check
```

Run `python scripts/run_pocketfinancer_pipeline.py check` when changing the app
profile, locks, pipeline config, or stage orchestration. Run targeted CUDA,
conversion, GGUF, and device checks only when the affected layer requires them;
report exactly which tier was and was not verified.

## Version control and pull requests

Use a short-lived branch and a focused pull request. Do not push development work
directly to `main`. Use a Conventional Commit title such as `feat:`, `fix:`,
`docs:`, `test:`, `refactor:`, `build:`, `ci:`, or `chore:`. Preserve unrelated
worktree changes and never weaken tests, privacy guards, or provenance checks to
make a change pass.

The pull request must state the engineering outcome, app/experiment impact,
verification commands and results, privacy/reproducibility considerations, and
known runtime-parity gaps. Create and update the pull request when credentials and
authorization permit; otherwise report the exact external blocker.
