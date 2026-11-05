# Cross-repository cleanup plan and handoff

Status: **coordination bootstrap; implementation/deletion audit is next**
Last checked: **2026-11-05**

This is the single temporary work plan for `pocket-financer-intelligence`,
`pocket-financer-android`, and `pocket-financer-ios`. It records the owner's
approved direction, not a claim that the target is implemented. Read the three
`AGENTS.md` files as well. For this cleanup, the target below supersedes older
roadmaps, experimental product choices, and blanket historical-code retention
requirements. Existing code/tests establish actual behavior, not future intent.
Privacy, source-data protection, branch protection, and release controls remain.

Update this document in place; do not create session diaries or parallel plans.
After completion, move lasting guidance into the few permanent docs and remove
this temporary plan. Older roadmaps are reference material pending consolidation,
not competing task lists.

## 1. Owner-approved target

### Scope and ownership

- Make all three repos lean around one current processing flow. Remove obsolete
  selectors, experimental pipelines, unused assets/dependencies, version adapters,
  scripts, tests solely for removed flows, and irrelevant Markdown documentation
  after their dependencies and useful unmerged work have been audited. Git history
  retains committed experiments; do not keep runnable historical pipelines merely
  for archival purposes. Current testing/evaluation is not obsolete baggage.
- Disposable development app databases and generated development datasets may be
  reset. Identify the exact artifacts before a reset. This does **not** authorize
  deletion of original SMS archives, human labels/corrections, provenance mappings,
  encryption keys, or their necessary backups. No reset is needed for the audit.
- Intelligence owns the shared design/contracts/prompts and Python evaluation,
  dataset/annotation, and improvement tooling. Android and iOS own their native
  intake, inference, validation, persistence, and UI. Native apps link contributors
  to the design; they do not call Python or depend on this repo at runtime.
- No shared UI framework, wholesale rewrite, automatic training-label promotion,
  publication, model deployment, or release is authorized by this plan.

### Product flow

```text
Receive/discover SMS or supported imported alert
  -> same-source identity / existing-outcome check
  -> narrow deterministic prefilter
  -> durable eligible work + per-operation settings snapshot
  -> advisory analysis
  -> one local SLM classification/extraction
  -> strict parsing, source grounding, exact money normalization
  -> account resolution + possible same-payment duplicate check
  -> Transactions / No transaction / Review
```

Platform intake may persist evidence earlier for crash safety; eligible evidence
must be durable before inference. Filtering must precede inference and behave
consistently across realtime intake, historical import, and manual scans.

- Prefilter confidently irrelevant standalone OTPs, ads, and similarly tested
  non-events without calling the model or creating Review. Uncertainty passes
  through. Missing regex matches are not proof of irrelevance; an `OTP` keyword
  alone cannot reject a completed transaction. Exclusion is not deletion from the
  phone's SMS inbox. Define minimal retention for excluded/`none` outcomes during
  implementation; do not invent a new retention policy during this bootstrap.
- The analyzer supplies optional evidence, not an extraction allowlist or an
  overriding semantic decision. The unchanged message is the grounding authority.
- The model returns `posted`, `none`, or `abstain`. `Review` is a host destination,
  not a model classification. The host never equates `posted` with permission to
  save without validation.
- Fields: required amount, currency, debit/credit direction, and account reference;
  optional counterparty/merchant. Explicit message currency overrides configured
  primary currency. Receipt/acquisition time is host-owned, not model-generated.
  Preserve exact minor-unit money and source-backed evidence.
- Automatically save complete, valid, uniquely resolved, non-duplicate results.
  Valid `none` creates no transaction or Review. Uncertainty, invalid extraction,
  unresolved accounts, and failures remain reviewable/retryable, not false `none`.
- First unknown account goes to Review. Owner confirmation can atomically reuse
  or create an account/alias and save the transaction. Later messages can resolve
  automatically. Missing counterparty alone must not block saving.
- Supported scope is one transaction per SMS. Do not build a multi-event extractor.
- Review shows original evidence and independently valid partial fields. Keep
  analyzer suggestions separate from model output; support source-backed owner
  correction and atomic confirmation. Corrections remain local feedback until
  explicit export/adjudication, not automatic approved training truth.

### Two duplicate checks, not one late check

1. **Same SMS before inference:** use reliable source identity and durable state
   to reuse the outcome or resume work. No unnecessary model call, second ledger
   entry, or duplicate Review case. Identical text alone is not reliable identity.
   An explicit owner-requested extraction retry is distinct from ingestion replay.
2. **Different SMS, possibly same payment after extraction:** compare grounded
   fields/resolved account and available timing/counterparty context. Flag credible
   uncertainty for Review rather than silently merge/delete. Same amount alone is
   insufficient. Offer a duplicate-versus-separate-payment resolution. Exact match
   rules and time windows remain to be specified/tested, not silently assumed.

Retain atomic writes and database uniqueness as final race/retry protection.

### Evaluation is a first-class deliverable

- Build one Python-based replica of the current end-to-end app behavior, with
  platform runtime adapters, datasets, isolated account/ledger/Review state, shared
  scoring, and reproducible configuration. Do not require app deployment for the
  normal prompt/model comparison loop.
- Android-target lane: retain useful local GGUF/llama.cpp and existing evaluation
  framework integration, including `lm-evaluation-harness` where appropriate.
  Audit reuse; do not mandate a new framework or preserve obsolete flows for it.
- Apple lane: use **Apple's official Python Foundation Models SDK on the Mac**.
  A Swift/Xcode runner is **not** the required evaluation architecture.
- Match each app's filtering, prompt/request, output protocol, generation settings
  where supported, parsing, grounding, normalization, accounts, duplicates, and
  routing. Test deterministic Python/Kotlin/Swift parity with the same supplied
  response and state; separate model calls need not produce identical responses.
  Document API differences rather than claiming a different protocol is parity.
- Include stateful sequences: unknown account, explicit simulated confirmation,
  subsequent auto-save, same-SMS replay, possible duplicate, interruption/retry.
  Simulated owner actions are not model successes. Define fresh run, resume,
  partial output, and changed-configuration rejection; keep per-run state isolated.
- Score **all input messages**, including filtered rows: false filter exclusions,
  avoided model calls, classification/field/grounding quality, correct/incorrect
  saves, missed transactions, Review rate/useful partial fields, account/duplicate
  behavior, failures, and host stage timings. Grounding alone does not prove the
  model selected the right amount or transaction semantics.
- Use a fixed labelled development cohort for tuning and a protected held-out
  cohort for quality claims. The 203-row regression fixture is not fresh gold.
  Keep private inputs/outputs local and ignored; do not use hosted inference.
- Improvement loop: baseline -> controlled change -> compare -> adopt in native
  app -> parity check. Desktop quality/timing is not phone speed, battery,
  background execution, or native storage/device acceptance evidence.

### Documentation end state

Keep a small useful set: README/getting started, contributor/agent rules, shared
architecture, evaluation guide, and necessary native build/device/privacy/release
instructions. Consolidate duplicates; remove obsolete plans/status reports and
fix links/tooling references. Review legal/license obligations separately. Do not
replace old document sprawl with a new document per session or version.

## 2. Verified starting points, not completion claims

The source observations below came from these **unmerged feature tips**, not
`main`. Preserve and inspect them during the audit:

| Repository | Starting checkout branch | Observed commit |
| --- | --- | --- |
| Intelligence | `codex/workbench-eval-set-filter` | `0fd9b8aece174f28f2b390d8d7fe99bc0ebc58fe` |
| Android | `codex/feat-native-sms-v3` | `54a8aa7ab9c0c4bc4c640c5534d7e637d27eecfd` |
| iOS | `codex/feat-native-sms-integration` | `050010e78afff3712750c67e206833df11170415` |

- Android `PipelineService` uses the automatic direct-extraction coordinator.
  `OnboardingService` and `HomeSyncManager` still apply old `SmsFilterPipeline`
  rules; the realtime `SmsParserWorker`/current coordinator do not use that same
  terminal prefilter. Filtering is **inconsistent**, not entirely absent.
- Android source-identity/atomic-save protections exist. The payment fingerprint
  includes exact receipt time, so it is not a complete separate-alert duplicate
  solution. Review account creation and partial-evidence handling are useful work.
- iOS's `SmsV4ProcessingCoordinator` is still review-only. Its
  `FoundationSmsExtractor` requests raw text; do not assume guided-schema parity.
- Intelligence `src/pocketfinancer_sms/evaluation.py` (`evaluate-extractor`)
  implements GGUF extraction/scoring, not the full stateful product flow.
  `apple_evaluation.py` (`evaluate-apple-fm`) has an official Python SDK adapter
  restricted to synthetic cases and a different guided protocol from the Swift
  app. Live Mac results were not verified in this session.
- `scripts/evaluate_lfm25_android_gguf.py` reflects an older Android contract.
  Audit useful runtime/framework pieces without treating that old flow as current.

No new app, model, emulator, or device verification supports this bootstrap.

## 3. Open PR inventory

Metadata checked through GitHub on 2026-11-05; fetch/recheck before acting.
Every implementation/guidance disposition is **pending diff and ancestry audit**.
PR titles and dependency arrows do not prove unique work or safe supersession.

| Repo / PR | Head branch @ observed SHA | Base | Scope |
| --- | --- | --- | --- |
| Intelligence [#5](https://github.com/pocket-financer/pocket-financer-intelligence/pull/5) | `codex/feat-sms-processing-foundation` @ `11c84903026b` | `main` | Foundation/workbench |
| Intelligence [#6](https://github.com/pocket-financer/pocket-financer-intelligence/pull/6) | `codex/slm-primary-sms-extraction` @ `4c80dd5cb911` | `main` | Direct extraction/routing |
| Intelligence [#7](https://github.com/pocket-financer/pocket-financer-intelligence/pull/7) | `codex/sms-model-quality-eval` @ `bc5f32a10c2f` | #6 branch | Evaluation lanes |
| Intelligence [#8](https://github.com/pocket-financer/pocket-financer-intelligence/pull/8) | `codex/workbench-eval-set-filter` @ `0fd9b8aece17` | #7 branch | Workbench saved-message sets |
| Intelligence [#9](https://github.com/pocket-financer/pocket-financer-intelligence/pull/9) | `codex/docs-session-bootstrap` @ `858a7b0cf6eb` | #8 branch | Access/session guidance |
| Android [#40](https://github.com/pocket-financer/pocket-financer-android/pull/40) (draft) | `codex/feat-native-sms-integration` @ `cd4f135ed4c7` | `main` | Earlier integration |
| Android [#42](https://github.com/pocket-financer/pocket-financer-android/pull/42) | `codex/feat-native-sms-v3` @ `54a8aa7ab9c0` | `main` | Automatic routing/review |
| Android [#45](https://github.com/pocket-financer/pocket-financer-android/pull/45) | `codex/docs-session-bootstrap` @ `e118afa3e015` | #42 branch | Access/session guidance |
| iOS [#8](https://github.com/pocket-financer/pocket-financer-ios/pull/8) (draft) | `codex/feat-native-sms-integration` @ `050010e78aff` | `main` | Native integration |
| iOS [#9](https://github.com/pocket-financer/pocket-financer-ios/pull/9) | `codex/docs-session-bootstrap` @ `93dd70c7ce22` | #8 branch | Access/session guidance |

Leave Android dependency PRs #44/#46 and iOS dependency PR #10/release PR #3
separate. Never merge a release PR or publish as a cleanup side effect.

For each relevant PR, record retain/update, consolidate, or supersede with the
exact destination commits/files and reason. Inspect incremental diffs against its
own base and ancestry/overlap with other PRs; check review comments and checks.
Do not merge everything just to simplify the graph. Do not discard useful work
by considering only `main`. Close/delete nothing until preservation and the
integration strategy are reviewed. Keep focused PRs; no direct pushes to `main`,
force pushes, history rewriting, or unapproved publication.

## 4. Work checklist and next bounded task

- [x] Agree on product, evaluation, cleanup, and documentation expectations.
- [x] Record initial source observations and open-PR metadata.
- [ ] **Next: audit relevant PR diffs/ancestry and produce the file-level cleanup map.**
- [ ] Review the proposed integration branches, PR dispositions, reset scope,
      documentation consolidation, and acceptance tests with the owner.
- [ ] Implement the selected flow and Python evaluation in bounded, tested slices;
      align native platforms and remove superseded paths without deleting useful work.
- [ ] Complete docs/dependency cleanup, parity checks, and platform verification;
      reconcile PRs and retire this temporary plan.

The next session is an **audit/proposal**, not authorization for implementation,
resets, merges, PR closure, or deletion. Deliver in this document:

1. Updated PR graph, unique/useful work, and proposed disposition for each PR.
2. Concrete keep / simplify-consolidate / remove rows covering code, tests,
   prompts/configuration, runtime/framework integrations, dependencies, and `.md`
   files, with paths, reasons, dependents, and destination where applicable.
3. Proposed integration baseline for each repo, including the existing guidance
   PRs and these new handoff PRs. Identify overlaps rather than blindly cherry-pick.
4. A small first implementation slice and its acceptance checks. Flag undecided
   details (duplicate matching, retention, runtime/API mismatches), without
   reopening the already-approved direction or silently inventing requirements.

## 5. Session checkpoint

Bootstrap environment: sibling checkouts under `/home/ubuntu/pocket-financer` on
Linux, not the owner's WSL/Windows or Mac/device verification lanes. Do not infer
that tools, models, or private datasets from those environments are present here.

All three bootstrap branches are named `codex/docs-cleanup-handoff`, created from
refreshed `origin/main` solely for small docs-only PRs. **The working checkouts now
contain main-based code, not the feature tips in section 2.** The earlier branches
remain untouched. Audit them explicitly via refs/diffs or deliberate checkouts;
this docs bootstrap does not choose the implementation integration baseline.

| Repository | Bootstrap base (`origin/main`) | Handoff PR |
| --- | --- | --- |
| Intelligence | `7850964` | Pending creation |
| Android | `a58fcad` | Pending creation |
| iOS | `c79b7b5` | Pending creation |

Changes: this plan, a pointer in each `AGENTS.md`, and the intelligence docs index.
No runtime code, old document deletion, data reset, frozen asset, release file,
or existing feature PR was changed. No merge is part of this checkpoint.

Verification in this Linux bootstrap:

- `git diff --cached --check` passed in all three repos.
- `python3 scripts/check_repo_safety.py` passed in intelligence after staging the
  new document (tracked pathnames only; no private dataset was opened).
- A local Python check passed for the exact docs-only file set, main-based branch
  bases, balanced code fences, and all 20 local Markdown links in the changed docs,
  including the two sibling-repository pointers.
- The normal intelligence environment/test gate is **unavailable**, not passed:
  `source scripts/activate_wsl.sh` rejects generic Linux (native WSL2 required);
  `python3 -m ruff --version` and `python3 -m pytest --version` report missing
  modules. No dependency installation or environment-policy bypass was attempted.
- Android Gradle, iOS Xcode, model inference, emulator/device, and full code-test
  suites were not run for this docs-only bootstrap. CI results must be checked
  separately and must not be represented as local results.

Publication: handoff PRs pending creation. Git fetch and authenticated push
preflights passed for all repos. `gh` is absent; GitHub API access can be used
without exposing credential-helper secrets.

At each handoff, replace this checkpoint with current branches/commit references,
PR links, changes, exact checks/results and unavailable lanes, blockers, and the
next bounded task. Do not store private data or credentials here. Read Git status
and live PR state first; a checkpoint is not a substitute for verification.

## 6. Resume prompt

> Read all three repositories' AGENTS.md files and
> pocket-financer-intelligence/docs/CLEANUP_PLAN.md. Verify working trees,
> branches, remote refs, and relevant open PRs. Continue the next bounded task:
> audit unmerged PR diffs/ancestry and propose a concrete keep/change/remove map
> covering code and Markdown docs. The current handoff branches are main-based;
> inspect the recorded feature branches, not just the current checkout. Preserve
> useful evaluation/workbench/native work and keep Apple evaluation on its official
> Python SDK. No implementation, deletion, reset, merge, PR closure, force-push,
> or release during this audit. Update the single plan with findings, verification,
> proposed integration paths, and the next step for owner approval.
