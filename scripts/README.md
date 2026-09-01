# Command map

## Active SMS processing and workbench

Pure-Python foundation and local review commands can run from the checkout that
contains the ignored private archive:

| Command | Purpose |
|---|---|
| `python scripts/run_sms_processing.py build-corpus` | Build/reuse the immutable 17,830-row canonical private run with explicit INR configuration. |
| `python scripts/run_sms_processing.py init-workbench` | Import the current canonical manifest into the crash-safe local SQLite workbench. |
| `python scripts/run_sms_processing.py serve-workbench` | Serve the token-protected workbench on `127.0.0.1` only. |
| `python scripts/run_sms_processing.py backup-workbench` | Create a mode-0600 SQLite backup and SHA-256 manifest. |
| `python scripts/run_sms_processing.py export-workbench` | Export submitted/adjudicated canonical labels into a reproducible hash-bound private directory. |

The corpus command prints aggregate counts only. The UI and all generated state
remain below ignored `PRIVATE_DATA/sms_processing`. Never tunnel, screen-share, or
open private rows in browser developer tools intended for capture. Synthetic HTTP
tests are the only supported UI smoke evidence in Git/CI.

## Historical model research environment

Run the following commands inside WSL2 after activating the environment from the canonical
checkout with `source /home/tojinotzenin/pF_slm_selection/scripts/activate_wsl.sh`.
Then enter the checkout that owns the code. Keep linked worktrees outside the
canonical repository and generated-artifact roots, and use only synthetic data
there. The privacy guard intentionally rejects private roots that resolve outside
the active checkout. Run private workflows from the canonical checkout after the
change is merged; never symlink or copy `PRIVATE_DATA` into a worktree.

## Historical PocketFinancer model pipeline

`scripts/run_pocketfinancer_pipeline.py` reproduces the previous app-facing model
research workflow. The default declaration is
`configs/pipelines/pocketfinancer-lfm2.5-350m.json`. The parallel
`configs/pipelines/pocketfinancer-lfm2.5-2.6b-base.json` declaration applies the
same workflow to the pinned LFM2.5-2.6B-Base checkpoint and must be selected
explicitly with `--config`. Both declarations expose these stages:

| Stage | Purpose |
|---|---|
| `check` | Report local readiness and exact inputs without modifying anything |
| `plan` | Print every command as an argv array |
| `build-data` | Materialize app-filtered, source-grounded private train/dev rows |
| `evaluate-base-hf` | Evaluate the locked, untuned HF checkpoint before adaptation |
| `train` | Train completion-only LoRA with the exact PocketFinancer messages |
| `evaluate-hf` | Evaluate the trained HF adapter using the app prompt and prefilter |
| `merge` | Merge the chosen adapter into the HF base |
| `convert` | Produce reference and deployable GGUF files |
| `evaluate-gguf` | Evaluate the deployable artifact under the app runtime profile |

Execution order is `build-data`, `evaluate-base-hf`, `train`, `evaluate-hf`,
`merge`, `convert`, then `evaluate-gguf`. Use `--dry-run` with an execution stage
to inspect only that command. `all` exists for an intentional end-to-end run, but
stage-by-stage execution is preferable while developing because each artifact can
be inspected before the next expensive step.

When thinking mode is enabled, the HF and GGUF evaluators retain raw reasoning
only in `samples.jsonl` beneath ignored local `RESULTS/` directories; aggregate
metrics do not contain it. Evaluation timing is measured on the WSL host, not on
an Android device, and is not Android runtime-performance evidence.

## Environment and safety

| Command | Purpose |
|---|---|
| `python scripts/verify_gpu.py` | Verify CUDA/PyTorch and local runtime readiness |
| `python scripts/verify_lfm25_backward.py` | Run a real short LFM backward-pass probe |
| `python scripts/probe_lfm25_lora_memory.py --help` | Inspect the one-backward-pass LoRA capacity gate; it is not training or quality evidence |
| `python scripts/check_repo_safety.py` | Check that private/generated artifacts cannot be committed |
| `bash scripts/setup_wsl.sh` | Bootstrap the pinned native WSL environment |

## Current data builders

| Command | Purpose |
|---|---|
| `python scripts/review_lfm25_blinded_test.py export` | Freeze all test rows into a reviewer-blind, test-only local package |
| `python scripts/review_lfm25_blinded_test.py export --source-prefill unambiguous` | Initialize the distinct candidate-assisted blinded package before its first workbench launch |
| `python scripts/review_lfm25_blinded_test.py validate` | Validate complete or resumable partial human annotations without writing outputs |
| `python scripts/run_lfm25_annotation_workbench.py blinded --reviewer <stable-id> --batch-size 50` | Review the frozen test package in a strictly local browser and durable blinded-test database |
| `python scripts/run_lfm25_annotation_workbench.py training --reviewer <stable-id> --pool <private-jsonl> --batch-size 50` | Curate an explicit train/dev-only pool in an isolated local database |
| `python scripts/run_lfm25_annotation_workbench_smoke.py` | Launch the real loopback UI with three invented rows and temporary state; no private loader or export is reachable |
| `python scripts/run_lfm25_annotation_workbench.py export-training --reviewer <stable-id> --pool <private-jsonl>` | Export reviewed training curation plus an aggregate-only report without overwriting nonempty outputs |
| `python scripts/review_lfm25_blinded_test.py import --workbench-db PRIVATE_DATA/lfm25/annotation_workbench/blinded_test.sqlite3` | Write completed test labels only after the durable workbench and required delayed-QC gate pass |
| `python scripts/evaluate_lfm25_components.py --input DATA/annotation_component_v1_synthetic.jsonl --dry-run` | Validate the invented paired fixture and print aggregate component metrics only |
| `python scripts/build_lfm25_private_sft_v2.py` | Rebuild the private source-grounded direct SFT set |
| `python scripts/build_lfm25_candidate_sft.py` | Convert grounded direct rows to candidate-selector rows |
| `python scripts/build_lfm25_candidate_curriculum.py` | Build low-weight semantic curriculum and mixed training data |
| `python scripts/audit_lfm25_candidate_coverage.py` | Audit deterministic candidate coverage without training |

The workbench binds only to `127.0.0.1`; do not tunnel or share its URL. It writes
state and rolling backups only below ignored `PRIVATE_DATA/lfm25`. Relaunch with
the same mode, reviewer, database, batch size, and (for training) pool to resume.

Open only the `http://127.0.0.1:<port>/` URL printed by the process. Opening
`lfm25/annotation_assets/index.html` as a `file://` page bypasses the stylesheet,
JavaScript, session, and row API, so it remains unstyled with no SMS and `0 / 0`.
Use the synthetic smoke command above for UI/manual-browser verification; it
stores its invented SQLite state under a temporary directory and removes it on
shutdown.

Source prefill is off unless explicitly selected. Assisted runs use
`--source-prefill unambiguous`, are recorded as
`human_verified_candidate_assisted`, and require a complete separate artifact
lineage. First run assisted `export`, then repeat the flag on `validate`, every
workbench launch/resume, and `import`. The assisted review JSONL, internal map,
metadata, database, reviewed manifest, and import report never serve as the
unaided projection or outputs. A reviewer must verify every suggested field and
exact span; assisted and unaided results are methodologically distinct and must be
reported separately. See the operating guide for exact commands and paths.

Training mode's generic `active_learning` filter deterministically prioritizes the
ten documented category classes from the frozen pool/order, never from test. It
hides the applicable category reasons until the first complete blind label; after
completion the UI can show those local reasons, while proposal details still
require an explicit **Reveal proposal** action.
After all primary rows are complete and uncertainty is resolved, use **Start
delayed QC pass** in the UI; a later relaunch of the same database resumes that QC
session. Pending QC hides the previous annotation and event history; actionable
initial/QC/adjudication history appears only after that blindness lifts. See the
[local workbench operating guide](../docs/guides/LOCAL_ANNOTATION_WORKBENCH.md) for
the ten category definitions, 50-row attention boundary, recovery command, and
final gates.

For an unaided run, give reviewers only `blinded_test_review.jsonl`; for an
assisted run, give them only `blinded_test_candidate_assisted_review.jsonl`. The
matching ID map and provenance metadata remain internal, and files from the two
policies must never be mixed. Import never edits the frozen
`split_manifest.jsonl`: it writes the policy-selected reviewed manifest and
aggregate report separately.
It now requires the matching completed workbench database, no unresolved rows,
and every required delayed-QC row to have passed. Existing nonempty outputs require
an explicit `--force`. Package metadata and the import report are committed last
as validity markers; a detected source change rolls the group back to its prior
state.

The final blinded reviewed manifest and reviewed training export preserve a
`human_annotation_workbench` object with the generic annotation and event
provenance. Private component-evaluation builders may pass its `annotation` field
to `lfm25.component_evaluation.adapt_workbench_annotation(...)`; raw source rows,
annotations, spans, event hashes, and per-row results must stay out of console
output and Git.

For each completed row, set `decision` to `transaction` or `not_transaction`,
then fill `reviewer` and an ISO-8601 `reviewed_at` timestamp with a timezone.
Transactions also require a numeric `amount`, `type` (`debit` or `credit`), and
nonempty `account`; `counterparty` may be null. Non-transactions keep all four
extraction fields null. Leave every annotation field null on untouched rows;
`notes` is optional only after a decision is complete.

Never put raw messages, spans, notes, row-level output, or screenshots in a
terminal transcript, issue, pull request, or documentation. The workbench and
component evaluator intentionally emit aggregate/status JSON only; keep their
private inputs and outputs local.

The filenames preserve compatibility with historical manifests. Dataset status and
actual semantic versions are recorded in [the experiment catalog](../docs/experiments/EXPERIMENT_CATALOG.md).

## Legacy private source archive acquisition

Two root utilities preserve the old, local-only acquisition lineage from an
iTunes backup to the source export accepted by private-data builders. They are
not validated stages of the supported PocketFinancer pipeline:

| Command | Purpose |
|---|---|
| `python find_sms_db.py` | Legacy/unvalidated helper: from a controlled backup parent, choose the newest entry and copy its Messages database to `sms.db` |
| `python export_sms.py [path/to/sms.db]` | Export recoverable iOS message text to `all_sms.json` and `all_sms.csv` |

`find_sms_db.py` does not validate directory candidates before choosing the newest
entry. Do not run it from the repository root or an uninspected directory. The
database and exports contain private data and must remain local and ignored.

## Underlying training and evaluation commands

| Command | Purpose |
|---|---|
| `python scripts/train_lfm25_lora.py --help` | Low-level LoRA trainer; PocketFinancer is the default profile |
| `python scripts/evaluate_lfm25_candidate_hf.py --help` | Evaluate candidate selection and strict reconstruction |
| `python scripts/evaluate_lfm25_android_hf.py --help` | HF prompt/training proxy; not a llama.cpp/Android runtime proof |
| `python scripts/evaluate_lfm25_android_gguf.py --help` | Evaluate a deployable GGUF against the current runtime contract |
| `python scripts/merge_lfm25_lora.py --help` | Merge a selected adapter into its HF base |
| `bash scripts/convert_lfm25_gguf.sh --help` | Convert/quantize a merged model with the pinned toolchain |
| `python scripts/compare_lfm25_predictions.py --help` | Compare paired per-row predictions |

## Historical/general model slate

`bash scripts/evaluate.sh` and root `run_gguf_eval.py` are the older broad GGUF
benchmark. They remain useful for exploration but are not the default app-facing
pipeline.

The dated results and failure signatures from that work are preserved in the
[general GGUF benchmark history](../docs/history/GENERAL_GGUF_BENCHMARK_2026-04-25.md).

## Historical commands

`scripts/run_lfm25_experiments.py`, `evaluate_lfm25_hf.py`,
`evaluate_lfm25_gguf.py`, `build_lfm25_synthetic_sft.py`, and
`build_lfm25_public_candidate.py` reproduce older short-prompt/synthetic work. Do
not use them to claim current app parity. The runner intentionally rejects nonlegacy
prompt profiles and fails closed on stale cached provenance.
