# PocketFinancer Extraction V2 status

## Completed and reviewed: Phase C — Android baseline reproducibility and runtime instrumentation

Phase C captured a reproducible, source-only Android baseline without changing
either mobile application or the locked production profile.

- Added the immutable `pocketfinancer-android-552ffbdf-phase-c` baseline. Its
  verifier hashes 31 canonical Git blobs at Android
  `552ffbdfbd41773980aa249789b0cb508fdb19fd` and never hashes dirty working-tree
  files.
- Verified all 13 files in the unchanged production profile at
  `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`. The baseline commit is three
  descendants later. Only the profiled `PipelineService.kt` path changed, adding
  invocation-scoped filter, inference, token, cache/performance, and persistence
  observer events.
- Captured the six-stage filter, prompt assets, parser behavior, chat-template
  boundary, decode/runtime settings, five selectable fleet identifiers, current
  persistence/time semantics, and manual/historical/automatic UI stage behavior.
- Froze an aggregate-only runtime evidence contract and fail-closed validator.
  Host and Android-device captures have distinct provenance. Packages reject
  unexpected fields, row/message content, raw device identity, inconsistent
  counts, non-monotonic quantiles, and a baseline hash other than the exact Phase
  C manifest.
- Added an entirely invented host/device aggregate smoke package. Its values are
  schema examples, not measurements or product evidence.
- Recorded current gaps: binary-floating-point amount storage, no explicit
  timestamp-provenance field, no content-free Android aggregate export, no
  end-to-end/stage latency, no verified model-load timing, no peak RSS, and no
  battery instrumentation.

Phase C inspected no private SMS, sender, account, annotation, row-level
prediction, model prompt/output, app storage, device log, model weight, or
generated private artifact. It performed no training, download, CUDA, HF/GGUF
inference, Android build/deployment, release, production-default change, or Phase
D experiment. iOS remained read-only.

## Android source/profile result

The locked profile file remains unchanged with SHA-256
`b818397851bb03fc3365710fe8ddfe82cadfe896525d931b93dbb8a6653c9d3f`.
Its revision `a9b7df44` is a reproducible production-contract snapshot. Current
committed Android `552ffbdf` retains the same profiled filter, prompts, parser,
native runtime, model declarations, and defaults, but adds newer pipeline/UI
observability.

Phase C deliberately did not advance `pocketfinancer-android-current.json`. The
separate baseline manifest records both revisions and the exact one-path
relationship. This prevents a newer checkout from being silently described as the
older locked profile.

The Android checkout had 149 unrelated dirty entries at verification time. The
baseline verifier reported the count only. It read the named committed Git objects
and did not change, stage, commit, reset, stash, clean, or overwrite that worktree.
The iOS checkout also retained all of its unrelated changes and remained read-only.

## Runtime and persistence baseline

The current committed path is custom Kotlin/JNI llama.cpp `b9198` on Android CPU:
3,072 context tokens, `n_batch=512`, `n_ubatch=256`, automatic threads capped at
four, zero GPU layers, flash attention, no mmap, F16-or-Q8 KV cache, built-in GGUF
chat template with a fallback prompt, greedy temperature-zero generation, optional
default-off grammar, and prefix-session caching. Thinking variants use up to 1,024
thinking tokens plus 256 answer tokens; non-thinking variants use the direct
256-token answer path.

Current observer/runtime facts include filter disposition, model identity,
thinking/grammar flags, token budgets, thinking and JSON token deltas,
prompt-evaluation and generation milliseconds, generated tokens, token rate, and
prefix-cache attempted/hit/prefix counts. The UI maps these into filter,
cache/prompt, inference, encrypted-persistence, and terminal outcome stages.

These facts are source and per-operation observability, not selection-grade
aggregate phone evidence. `TransactionEntity` still stores amount as `Double`.
`SmsMessage.date` supplies the ledger timestamp, while source identity prefers
`date_sent` when present, but explicit timestamp provenance is not persisted.
Phase C records those gaps without implementing Semantic V2 storage or changing
application behavior.

## Evidence classes

- Source/static: 31 baseline blobs and 13 locked-profile blobs verified.
- Host: lightweight Python verification/tests only. No host runtime timing claim
  was made.
- Android device: not measured. A content-free `adb get-state` probe reported no
  attached device or emulator. Phase C did not start an emulator or run inference.

Host evidence cannot satisfy an Android-device gate. The invented aggregate
fixture cannot satisfy either a host or device product gate.

## Branches, bases, and commits

- Phase C SLM branch: `codex/extraction-v2-phase-c` from Phase B handoff
  `7eb22ec76a90c7dac72cac8af8665e71fef6dd6b`.
- Baseline and aggregate runtime instrumentation:
  `4ab91ec8ffb71f4531861211e674edca639a28f0`
  (`feat: capture Extraction V2 Android baseline`).
- Android read-only committed baseline:
  `552ffbdfbd41773980aa249789b0cb508fdb19fd`.
- Locked production-profile Android revision:
  `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`.
- iOS read-only base remained
  `04f770b235f860080ecd96adad1a5d011f3c2c2c`.

The final state/handoff commit records the immutable implementation commit above.
Its own final object ID is reported by the task that creates it because a commit
cannot contain its own hash.

## Frozen SHA-256 artifacts

Phase A and B hashes remain unchanged and are still recomputed by the program-state
test.

Phase C artifacts:

- Android baseline manifest:
  `9274e5a63524b46bb4149e11d5190bae4ebcfef15170a69af77ad050b31167fc`.
- Android baseline reference:
  `08ddcd4c4aceb59b9252b01538e2a941754b6ad010db586d0701d61ae7b9641d`.
- Android baseline CLI:
  `8ddf1fc242857b1b0a4513fa577bbb8895dec9c8cf7b23298516b443a2367086`.
- Aggregate runtime evidence contract:
  `710b05a6a84b1434633b533389d50dfd693170919ef0ee368eac9d390787e18f`.
- Aggregate runtime evidence reference:
  `e2b9e0ed21f00f86bc637023fe89891cd533d703b562bd909b3a71fc790960bc`.
- Aggregate runtime evidence CLI:
  `71e73b768779684e82612680b58f1c454f3af0c848437eaec3f682ecbf862ead`.
- Invented aggregate runtime fixture:
  `45eb91c0f184cd18f5e394c6abb29b027382f954ce0af2d64b96990ba083e3fb`.
- Phase C audit report:
  `49e66435a26f656f9d91fa031e0136838a7e4e75c261f3adf014e6c83b6146f7`.

The program-state test recomputes every Phase A, B, and C hash and fails on
unrecorded drift.

## Verification

- Entry hash lock:
  `pytest -q tests/test_extraction_v2_program_state.py` — 1 passed before branch
  or implementation changes.
- Phase C target:
  `pytest -q tests/test_android_baseline.py tests/test_android_runtime_evidence.py tests/test_android_profile_sync.py tests/test_android_contract.py`
  — 41 passed.
- Baseline verifier: 31 baseline blobs and 13 profile blobs verified; Android HEAD
  matched `552ffbdf` and the only profiled-path delta was `PipelineService.kt`.
- Invented aggregate package validator: passed with explicit, separate `host` and
  `android_device` evidence classes.
- `python scripts/check_repo_safety.py` — passed; it reported only the
  pre-existing publication-review exception for `DATA/extraction_ds.jsonl`.
- `ruff check .` — passed.
- `ruff check --select E4,E7,E9,F lfm25 scripts tests` — passed.
- `pytest -q` — 591 passed.
- `git diff --check` — passed.

`python scripts/run_pocketfinancer_pipeline.py check` was attempted as an
additional guard. It refused the pre-existing auto-discovered WSL Android clone at
`a6c8a11` because that checkout is not the locked profile revision `a9b7df44`.
No pipeline, production profile, model lock, or orchestration file changed in
Phase C; the provenance refusal was preserved rather than weakened. The explicit
Phase C baseline verifier passed against the user-named Android repository and
both required commits.

Only source/static and lightweight host verification ran. Android native builds,
model inference, phone latency/memory/battery, and iOS builds/devices were
intentionally not run.

## Program status and unresolved work

- The program definition of done is not achieved. No profile is selected, no
  Android runtime variant has a Phase H disposition, and no iOS profile is
  selected.
- Direct V2 and Candidate V2 remain unselected hypotheses. Phase C made no quality
  comparison.
- Protected or blinded scoring remains unauthorized. Private data rights,
  annotation/adjudication policy, and split governance still require separate user
  approval.
- Device-specific hard budgets remain absent and must be versioned before the
  applicable first protected device measurement.
- Android device evidence remains absent. Host/source evidence cannot replace it.
- Exact-money and timestamp-provenance storage, Semantic V2 adapters, migration,
  grounding, and production integration remain later-phase work.
- Neither mobile application contains a Semantic V2 production adapter or a
  manifest-bound default-off UAT candidate.

## Next allowed phase: D — Android prompt/output-protocol laboratory

Phase D may start only in a new task from the final Phase C handoff commit reported
by this task. It must:

1. read every applicable `AGENTS.md` completely and run the program-state hash-lock
   test before acting;
2. preserve every unrelated change in all three worktrees and keep iOS read-only;
3. keep Semantic V2, Workbench V2, the decision policy, and all Phase C baseline
   and runtime-evidence artifacts hash-locked;
4. bind every comparison to Android `552ffbdf` and the Phase C manifest;
5. compare Direct V2 and Candidate V2 independently for Qwen and Gemma using
   controlled, evaluation-only evidence; do not imply a universal protocol;
6. keep host, GGUF-host, and Android-device evidence distinct and retain missing
   device measurements as gaps;
7. use only invented/committed synthetic material unless the user separately
   authorizes private data rights, sources, adjudication, and split governance;
8. not train or download models, run CUDA, protected scoring, or private-data
   inference without separate explicit authorization; and
9. not select Direct V2, Candidate V2, a model, or a profile, not modify either
   mobile application or production defaults, and not proceed into Phase E.

Copy-paste prompt for the next task:

```text
Continue the PocketFinancer Extraction V2 program with Phase D only: the Android prompt/output-protocol laboratory. Start from branch codex/extraction-v2-phase-c at the final Phase C handoff commit reported by the Phase C task, using the finalized plan, hash-locked program state, frozen Semantic V2/Workbench V2/decision-policy artifacts, and the Phase C Android baseline/runtime-evidence artifacts recorded in docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md and configs/programs/pocketfinancer-extraction-v2.json. Read every applicable AGENTS.md completely and run pytest -q tests/test_extraction_v2_program_state.py before acting. Verify the expected SLM, Android, and iOS commits and preserve every unrelated working-tree change; do not reset, stash, clean, overwrite, stage, or commit the extensive unrelated Android/iOS changes. Keep iOS read-only. Bind all work to Android 552ffbdfbd41773980aa249789b0cb508fdb19fd and the Phase C baseline manifest. Implement Phase D only with controlled, evaluation-only comparisons of Direct V2 and Candidate V2 independently for Qwen and Gemma; do not imply a universal protocol. Keep host, GGUF-host, and Android-device evidence explicitly separate, and do not treat missing device evidence as a pass. Use only invented/committed synthetic data unless I separately authorize private data rights, sources, annotation/adjudication, and split governance. Do not inspect, print, modify, or transmit private SMS or annotation rows; do not train or download models, run CUDA, protected scoring, or private-data inference without separate explicit authorization. Do not select Direct V2, Candidate V2, any model, or any profile; do not modify Android/iOS production code or defaults; and do not proceed into Phase E. Review the work, run all relevant lightweight and Phase D checks, make focused local Conventional Commits in the SLM repository only, and update the hash-locked program state and handoff with exact commit IDs and the next allowed phase. Do not push or open a pull request.
```
