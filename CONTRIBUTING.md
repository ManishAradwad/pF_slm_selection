# Contributing to PocketFinancer SLM Selection

This repository is a privacy-sensitive ML research and engineering workspace. A
useful contribution must be reproducible, honest about evaluation boundaries, and
compatible with PocketFinancer Android's actual inference path.

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
python -m ruff check --select E4,E7,E9,F lfm25 scripts tests
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
