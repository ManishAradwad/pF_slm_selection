# Contributing to PocketFinancer SLM Selection

This repository is a privacy-sensitive ML research and engineering workspace. A
useful contribution must be reproducible, honest about evaluation boundaries, and
compatible with the active shared SMS contracts. Legacy Android-profile work must
still reproduce the pinned app path accurately.

Start with [AGENTS.md](AGENTS.md), then read the
[documentation map](docs/README.md) and [command map](scripts/README.md).
Executable code and versioned profiles outrank prose when they disagree.

## Development environments

### Lightweight checks

Documentation, pure-Python logic, and most tests need no GPU or model download.
A clean lightweight environment can use `requirements-ci.txt`:

```bash
python3.11 -m venv /tmp/pf-slm-ci
source /tmp/pf-slm-ci/bin/activate
python -m pip install -r requirements-ci.txt
python scripts/check_repo_safety.py
python -m ruff check .
python -m ruff check --select E4,E7,E9,F lfm25 src scripts tests
python -m pytest -q
```

Torch-dependent tests skip in this tier. Do not add Torch or model libraries to CI
merely to avoid an appropriate runtime skip.

### Full WSL/CUDA environment

Training and local inference use the native WSL2 checkout and its prepared Python
3.11 environment:

```bash
cd /home/tojinotzenin/pF_slm_selection
bash scripts/setup_wsl.sh
source scripts/activate_wsl.sh
python scripts/verify_gpu.py
python scripts/run_pocketfinancer_pipeline.py check
```

`setup_wsl.sh` can install/download dependencies; inspect it before use in a new
environment. Model assets and private artifacts remain local and ignored.

## Making a change

1. Update from `origin/main` and create a short-lived, descriptive branch.
2. Inspect the complete affected path before editing.
3. Keep app-facing behavior in a versioned contract or pipeline declaration.
4. Add focused tests and update the relevant current documentation.
5. Run the lowest sufficient verification tier plus `check_repo_safety.py`.
6. Commit with a Conventional Commit subject and push the branch.
7. Open a focused pull request targeting `main`; do not bypass failing checks.

Do not combine a mechanical repository reorganization with new training behavior
unless the dependency is unavoidable. Preserve provenance-compatible paths or add
explicit shims/migrations.

### Linked-worktree hygiene

Keep development worktrees outside the canonical repository and outside ignored
generated roots such as `TRAINING_ARTIFACTS/`. A suitable layout is
`/home/tojinotzenin/worktrees/pF_slm_selection/<branch-name>`. Before editing or
opening VS Code, verify the folder rather than relying on the window title:

```bash
git worktree list
git rev-parse --show-toplevel
git branch --show-current
git status --short
```

Open the exact `--show-toplevel` directory in the WSL-connected VS Code window.
Uncommitted changes belong to that worktree; do not delete, reset, or move them
merely because another worktree shows a different branch. Remove a worktree only
after its status is clean and its commits are preserved on the intended branch.

Use linked worktrees only with synthetic fixtures until their changes are merged
into the canonical checkout. The runtime privacy guard intentionally rejects a
`PRIVATE_DATA` symlink whose target resolves outside that checkout. Never symlink
or copy private data into a development worktree or a synced/shared location.

## Verification by change type

| Change | Minimum verification |
|---|---|
| Documentation/config only | safety check, Ruff, unit tests, `git diff --check` |
| Pure Python behavior | focused tests plus full lightweight suite |
| Android profile/pipeline | full suite plus pipeline `check` and source-hash verification |
| Training/data builder | focused invariants, smoke materialization/backward pass, full suite |
| Merge/conversion/GGUF | locked toolchain checks, load smoke, paired prediction comparison |
| Deployment claim | GGUF parity plus instrumented target-phone measurement |

Record exact commands and outcomes. A skipped CUDA or phone check is acceptable
when irrelevant, but it must not be silently represented as passed.

## Data and result changes

Never include raw/private examples in tests, issues, pull requests, screenshots, or
logs. Aggregate reports may contain counts, hashes, and non-identifying run
metadata. Per-row result files, adapters, checkpoints, GGUFs, private manifests,
and candidate datasets stay outside Git.

Canonical SMS manifests, annotation-workbench databases, lock files, backups, reviewer projections,
training-curation exports, notes, spans, and row-level component inputs or outputs
are private artifacts under ignored `PRIVATE_DATA/sms_processing` (or historical
`PRIVATE_DATA/lfm25`). Run the UI only on its
`127.0.0.1` URL; never tunnel it, screen-share a private review, or move state into
a synced folder. Do not diagnose a row by printing its SMS, sender, annotation,
notes, or proposals to the console. Use opaque local IDs and aggregate counts only.

Treat every annotation-assistance policy as a distinct private artifact lineage.
Do not reuse or copy a review JSONL, internal map, metadata file, workbench DB,
reviewed manifest, or import report between unaided and candidate-assisted runs.
Initialize and operate each lineage only with its matching policy flag.

Tests, documentation, demos, and any UI image must use wholly invented messages and
identities. Never capture or attach a screenshot of a real review row, even when
the surrounding issue or pull request is private.

For annotation UI development, run
`python scripts/run_lfm25_annotation_workbench_smoke.py` and open the printed
loopback HTTP URL. Never open `lfm25/annotation_assets/index.html` directly: a
`file://` page cannot load the authenticated server APIs and therefore shows no
SMS. The smoke launcher uses only fixed invented rows and temporary state, making
it the appropriate source for UI screenshots and manual browser checks.

When adding or revising an experiment:

- define train/dev/test roles and leakage boundaries;
- distinguish human-gold, consensus-silver, grounded-silver, and synthetic labels;
- freeze selection criteria before opening a fresh test;
- report whole-pipeline and transaction-only metrics separately;
- report raw-output validity separately from app fail-closed interpretation;
- label HF, host GGUF, and Android-device measurements accurately; and
- update `docs/experiments/EXPERIMENT_CATALOG.md` with the result's trust status.

## Pull requests

Use the repository template. A good PR explains:

- what became possible or safer;
- which contract, data, training, evaluation, or runtime paths changed;
- what was deliberately left unchanged;
- exact verification evidence;
- privacy, reproducibility, compute, and compatibility implications; and
- any decision still blocked on human review, licensing, or device access.

This repository has no automatic model or dataset release path. Merging code does
not authorize publishing data, weights, or application artifacts.
