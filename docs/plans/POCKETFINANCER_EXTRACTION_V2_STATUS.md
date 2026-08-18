# PocketFinancer Extraction V2 status

## Completed and reviewed: Phase B — Workbench V2, evaluation packages, and frozen decision policy

- Added the frozen `pocketfinancer_workbench_v2` contract and dependency-free
  Python reference for local annotation-package creation, Semantic V2 validation,
  sender/template split isolation, independent adjudication, provenance binding,
  and export-scoped categorical anonymization.
- Added aggregate-safe Semantic V2 evaluation packages. A profile-specific parser
  may supply a valid mapped Semantic V2 record or an explicit invalid result; the
  independent evaluator validates every claimed record, fails malformed records
  closed, scores all required semantic fields, reports automatic-post precision
  and useful coverage, and emits a zero-content failure taxonomy.
- Added a local CLI that restricts raw and row-level inputs/outputs to ignored
  `PRIVATE_DATA/`, restricts aggregate reports to ignored `RESULTS/`, and prints
  structural or aggregate-safe values only. Its one public-input exception is the
  exact committed invented fixture behind an explicit flag.
- Froze decision-policy version 1 before any protected evaluation. It declares
  Wilson 95% confidence intervals; sample denominators; semantic, precision,
  coverage, invalid-output, and fail-closed thresholds; seven critical safety
  categories with zero automatic-post tolerance; provenance, baseline, operational,
  and platform-budget gates; material tie order; and explicit no-selection rules.
- Kept the comparison unit to one platform and one runtime variant. Direct V2 and
  Candidate V2 remain unselected hypotheses; neither is implemented or declared a
  universal protocol.
- Added an entirely invented four-row annotation/evaluation smoke package with
  independent synthetic reviewers, adjudication, split groups, valid and malformed
  Semantic V2 mappings, and all critical safety tags. Tests cover privacy path
  guards, raw-value removal, split leakage, adjudicator independence, provenance
  mismatch, malformed timestamp provenance, false automatic posting, aggregate
  output, and no-selection on insufficient evidence.

Phase B inspected no real SMS, annotation, sender, account, row-level prediction,
model, or generated private artifact. It performed no training, download, CUDA,
HF/GGUF inference, mobile build, deployment, release, or production-default change.
Android and iOS remained read-only.

## Frozen decision-policy summary

- Confidence: two-sided 95% Wilson score intervals; floors use the lower bound and
  ceilings use the upper bound.
- Minimum protected evidence: 1,000 rows, 250 gold-auto-post-eligible rows,
  400 predicted auto-posts, 250 applicable gold events per field, 400 invalid or
  fault-injection cases, 30 cases per critical safety category, and all 13 frozen
  conformance vectors.
- Lower-bound floors: scope/posting/cardinality 0.98; amount-currency, direction,
  account, and counterparty exactness 0.95; evidence validity and auto-post precision
  0.99; useful auto-post coverage 0.70; fail-closed rate 0.99.
- Upper-bound ceiling: invalid-output rate 0.02.
- Absolute safety: zero false automatic posts overall and zero automatic-post
  violations for ungrounded evidence, ambiguous currency, not-posted, none/multiple,
  non-INR, inexact-money, and missing-timestamp-provenance cases.
- A candidate also needs exact provenance, a locked protected split, a reproducible
  same-row baseline, same-tier operational evidence, and a versioned platform/device
  hard-budget attachment frozen before measurement. Host timing cannot satisfy a
  device gate.
- Passing candidates use the frozen material order: auto-post precision lower bound,
  coverage lower bound, semantic macro lower bound, latency p95, and memory p95.
  Lexical profile order never breaks a tie. Missing/failed evidence or an unresolved
  tie yields no selection.

The full normative artifact is
`configs/programs/pocketfinancer-extraction-v2-decision-policy.json`. The operator
guide is `docs/guides/EXTRACTION_V2_WORKBENCH.md`.

## Branches, bases, and commits

- Phase B SLM branch: `codex/extraction-v2-phase-b` from Phase A handoff
  `cd5fa75d03af6d3056847e63ff2cbddd9c382ff1`.
- Workbench foundation:
  `6365422c7775fbc0b6343d2ef077402b97e8e44f`
  (`feat: add Extraction V2 workbench foundation`).
- Evaluator and frozen policy:
  `e10a738603cf87656465046406bc8e444747592f`
  (`feat: freeze Extraction V2 evaluation policy`).
- Android read-only base remained `main` at
  `552ffbdfbd41773980aa249789b0cb508fdb19fd`.
- iOS read-only base remained `main` at
  `04f770b235f860080ecd96adad1a5d011f3c2c2c`.

Android and iOS both had extensive unrelated pre-existing working-tree changes at
entry. Phase B did not inspect their contents, stage them, or write either worktree.
The final state/handoff commit records the immutable implementation commits above;
its own final object ID is reported by the task that creates it because a commit
cannot contain its own hash.

## Frozen SHA-256 artifacts

Phase A artifacts remain unchanged and hash-locked:

- Normative plan: `6eccc6c7d5df21b4998079ff40c2d5c92e51af53a83a6cbad5fc15c29c7203f5`.
- Semantic V2 schema: `6c65b29543a85ca314f45620ed6300300d802ecab58d11371cc85af314a27bf8`.
- Semantic V2 reference: `ae0da990999035b8e04999d81854ed7ce4a95914a9079ff08eb50da03874c018`.
- Semantic V2 invented conformance fixture:
  `1bb7053d7b49066830eb0c08f546c8b3be5490e292f3497928c832f25b03f2fa`.

Phase B artifacts:

- Workbench V2 contract:
  `99f765f9d78411decaf0ab6cc6b72af388cf023d03725e2c71c536b33f4e1627`.
- Workbench V2 reference:
  `8daf92a7eba0b97ea61bb695d8a53719e1f56ac877e95a5d7f7afb3fcdefa65f`.
- Invented Workbench V2 fixture:
  `d0229c86f6ba8afa17ade9226e6ef4ed630a70c486821f4996f2a784ac8de3cb`.
- Frozen decision policy:
  `96dd8d2f4bff4cf8ec964dc668b05cbf75dd3ee11323bca623fe220fc55b49a4`.
- Semantic V2 evaluator:
  `559e6484e96f6cee85c97c158d529eb7acc12136f15c7c60351b4c58bb3547e3`.
- Local Workbench V2 CLI:
  `e311066aeb3dd39c6f9ccad9cc50b6b8ca68db394b981f7a2253d74e6422e407`.
- Workbench V2 operator guide:
  `ed6614ce413d4130316167074cbe5cea9be24b8974134a0dfc239e351bf5a43f`.

The program-state tests recompute these hashes and fail on unrecorded drift.

## Verification

- Entry hash lock:
  `pytest -q tests/test_extraction_v2_program_state.py` — 1 passed before any
  branch or implementation action.
- Phase B target:
  `pytest -q tests/test_workbench_v2.py tests/test_workbench_v2_cli.py tests/test_evaluation_v2.py`
  — 16 passed.
- `python scripts/check_repo_safety.py` — passed; it reported only the pre-existing
  publication-review exception for `DATA/extraction_ds.jsonl`.
- `ruff check .` — passed.
- `ruff check --select E4,E7,E9,F lfm25 scripts tests` — passed.
- `pytest -q` — 579 passed.
- `git diff --check` — passed.

Only the lightweight repository tier ran. Model downloads, training, CUDA,
HF/GGUF inference, Android builds/devices, and iOS builds/devices were intentionally
not run because they are outside Phase B.

## Program status and unresolved work

- The program definition of done is not achieved. No profile is selected, no
  Android runtime variant has a Phase H disposition, and no iOS profile is selected.
- Protected or blinded scoring remains unauthorized. The user must separately
  authorize data rights, annotation/adjudication policy, private sources, and the
  locked protected split before any real-data work.
- Device-specific hard budgets remain deliberately absent. Each applicable budget
  must be versioned and frozen before that platform's first protected device
  measurement; absence yields no selection.
- Android parser acceptance, latency, memory, battery, recovery, persistence, exact
  money/time storage, and UI evidence have not been regenerated under Semantic V2.
- Apple Foundation Models behavior remains OS/device dependent and unmeasured in
  this program.
- Neither mobile application contains a Semantic V2 production adapter or a
  manifest-bound default-off UAT candidate.

## Next allowed phase: C — Android baseline reproducibility and runtime instrumentation

Phase C may start only in a new task from the final Phase B handoff commit reported
by this task. It must:

1. read every applicable `AGENTS.md` completely and run the program state hash-lock
   tests before acting;
2. preserve all unrelated changes in every worktree and keep iOS read-only;
3. keep Semantic V2, Workbench V2, and the decision policy hash-locked, versioning
   any intentional change;
4. audit the current Android source/profile relationship and capture a reproducible
   baseline without changing locked production defaults;
5. label host and Android-device evidence separately and avoid private SMS or
   annotations without separate explicit user authorization; and
6. not select Direct V2, Candidate V2, or any profile, and not proceed into Phase D.

Copy-paste prompt for the next task:

```text
Continue the PocketFinancer Extraction V2 program with Phase C only: Android baseline reproducibility and runtime instrumentation. Start from the Phase B branch, commits, finalized plan, hash-locked program state, frozen Workbench V2 artifacts, and frozen decision policy recorded in docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md and configs/programs/pocketfinancer-extraction-v2.json. Read every applicable AGENTS.md completely and run the program state hash-lock tests first. Preserve all unrelated changes. Keep iOS read-only. Audit the current Android source/profile relationship and capture a reproducible baseline without changing production defaults. Keep host and device evidence distinct. Do not inspect private SMS or annotations without separate explicit authorization, do not select Direct V2, Candidate V2, or any profile, and do not proceed into Phase D. Run the relevant checks, make focused local Conventional Commits, and update the program state and handoff with exact commit IDs and the next allowed phase. Do not push or open a pull request.
```
