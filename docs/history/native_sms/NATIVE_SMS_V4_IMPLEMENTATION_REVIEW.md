# Native SMS v4 implementation review

> Historical evidence recorded on 2026-09-20. It is not the current plan and does
> not prove that previously observed emulator issues are fixed.

Date: 2026-09-20

Status: **app source implementation complete through mandatory step 8; Android
automated, emulator, and synthetic review-all verification passed; final Apple
and physical-device acceptance remains unverified**

This record continues the preserved post-step-5 checkpoint. It covers review
projection, atomic confirmation, native evidence selection, the Transactions
workflow, automated gates, Android emulator evidence, and the local synthetic
review-all trial. It does not authorize deployment or automatic persistence.

## Engineering outcome

Both native apps now have source-grounded v4 review implementations over the
additive `native-integration-v4` / `processing-config/4` contract. Historical
v1/v2/v3 routing and assets remain intact.

The implemented review boundary:

- accepts only posted `processing-result/3` results retained with `review` or
  `blocked` status;
- requires recognized receipt provenance and complete account-resolution and
  duplicate-assessment objects; validates canonical `vpa:`/`suffix:` alias
  keys, match counts, local account identity, alias hashes, source/event keys,
  and the recomputed transaction fingerprint;
- re-derives amount, direction, account, and counterparty from exact Unicode
  scalar evidence rather than trusting displayed or submitted normalized text;
- rejects floating-point contract integers, invalid spans, semantic/evidence
  mismatches, unsupported correction fields, and repeated corrections to one
  field;
- keeps receipt time immutable and projects its original provenance;
- rechecks account aliases inside the confirmation transaction, reuses one
  unique match, creates a new account and confirmed alias only when no match
  exists, and blocks ambiguity;
- blocks exact source/stable-event replay, preserves possible semantic
  duplicates for explicit user review, binds duplicate keys to the actual
  source and stable event, and permits an action replay only for the same
  review case;
- inserts the account/alias, transaction, revision, feedback, and resolved
  review state atomically.

No default account or balance behavior was introduced.

## Review and Transactions experience

Android Compose and iOS SwiftUI now provide the same intended interaction:

1. Processing and Needs Review appear above the confirmed ledger on
   Transactions.
2. Review shows sender context, reason codes, and read-only receipt time.
3. The complete SMS is selectable but not editable.
4. Amount, Direction, Account, and Counterparty remain visibly assigned.
5. One active field owns the platform's native selection handles while all
   assigned fields remain highlighted.
6. Clear and reselect operations create source-supported draft corrections.
7. Draft reload preserves exact selections, including deliberately cleared
   optional or required fields.
8. Confirmation, retry, and rejection return to the canonical Transactions
   flow. Invalid v4 results fail closed and never fall back to the legacy
   editable-time review screen.

The Android historical Reviews route remains a compatibility entry point. The
canonical actionable list is Transactions.

## Verification evidence

### Shared foundation and package parity

- `python scripts/check_repo_safety.py`: passed.
- `ruff check .`: passed.
- strict Ruff error selection: passed.
- `pytest -q`: **794 passed**.
- `git diff --check`: passed.
- Android and iOS v4 bundles each verified **44 artifacts / 45 files** against
  manifest SHA-256
  `0d3bf18f91d0a197c7bb56b5e082fd2851ce072f854452a9647c52d45b3433d8`.

### Android

The final repository gate passed:

```text
gradlew testDebugUnitTest lintDebug assembleDebug
        :app:compileDebugAndroidTestKotlin --no-daemon
BUILD SUCCESSFUL — 366 tasks
```

Focused in-memory Room coverage proves that confirmation creates exactly one
account/alias, transaction, revision, and feedback event; preserves receipt
time; makes same-review replay idempotent; restores deliberately cleared
required-field drafts; reuses one uniquely matched owned account; and rolls
back every confirmation effect for invalid grounding, ambiguous aliases,
mismatched duplicate identity, or duplicate field corrections. Producer and
parser tests also pin the canonical alias-key representation and the
`possible_duplicate` contract status. Emulator accessibility coverage verifies
that the complete evidence text exposes selection semantics without editable
text semantics.

Pixel 9 AVD, Android API 15:

- app instrumentation: **14 passed**;
- data recovery instrumentation: **4 passed**;
- pipeline package instrumentation: **2 passed**;
- total: **20 passed, 0 failed**.

This also serves as the local synthetic review-all trial: only synthetic SMS and
account data were used, the review-only confirmation path was exercised through
the domain/store boundary, native UI and accessibility behavior was covered by
separate emulator instrumentation, and no telemetry or remote inference was
enabled.

### iOS

Swift/XCTest coverage is present for grounded projection, strict integer and
status parsing, canonical account-alias keys and hashes, duplicate fingerprints
and source identity, Unicode scalar corrections, account creation/reuse,
atomic confirmation, review-bound replay idempotency, immutable receipt time,
migration, and recovery. Independent store actors use deterministic account and
alias identifiers so concurrent confirmation converges instead of creating two
accounts for one unmatched alias. The app source includes the native selectable
evidence view and the Transactions navigation flow. Scoped static safety and
whitespace checks pass.

### Acceptance matrix

| Area | Android | iOS |
|---|---|---|
| Contract producer/parser and atomic store | Implemented; JVM tests passed | Implemented; source/static checks only |
| Native review/Transactions UI | Implemented; emulator tests passed | Implemented; Xcode execution pending |
| Migration and interruption recovery | Automated tests passed | XCTest present; execution pending |
| Release build | Debug gate passed | Xcode 26 Release pending |
| Physical-device behavior | Pending | Pending |

This Windows/WSL host has no `swift`, `swiftc`, `xcodebuild`, `xcrun`, iOS
simulator, or iPhone connection. Therefore none of the following is claimed as
passed:

- Swift 6 compilation;
- XCTest or XCUITest execution;
- V6-to-V7 migration execution;
- Xcode 26 Release build;
- simulator or physical-iPhone runtime verification.

These checks require an Xcode 26 macOS runner and remain part of final
acceptance, not implementation work that can be completed on this host.

## Device and rollout boundary

Only the Android emulator was connected. No physical Android phone or iPhone
was available, so latency, memory, local-model behavior, background recovery,
selection handles, screen-reader behavior, and restart behavior still require
truthful physical-device evidence on both platforms.

Automatic persistence remains disabled. Every valid posted v4 result stays in
review. No private SMS, sender, account identifier, model output, or per-row
prediction was added to fixtures, logs, telemetry, or a hosted service. No
private data was moved between checkouts.

## Acceptance disposition

Mandatory execution steps 1–8 are implemented. Step 9 is complete for the
Android emulator only. Step 10 is complete as an Android synthetic local trial;
its Apple execution remains dependent on the missing macOS environment.

The remaining release blockers are external validation tasks:

1. run Swift 6 unit, migration, UI, and Release gates with Xcode 26;
2. run the iOS simulator flow;
3. capture aggregate, non-sensitive evidence on a physical Android phone and
   iPhone;
4. fix any platform-specific defect those runs expose, then update this record.

Until those checks pass, the implementation must not be described as fully
accepted for release or as evidence for enabling automatic persistence.
