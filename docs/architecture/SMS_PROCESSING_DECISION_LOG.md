# SMS Processing Decision Log

Status: **active**
Current decision reconciled: 2026-09-22

## Current selected design

PocketFinancer uses one local SLM as the central classifier and extractor. The
deterministic analyzer supplies advisory evidence only. The host strictly parses
and grounds Unicode-scalar spans, normalizes exact money, resolves accounts,
assesses duplicates, owns persistence/recovery, and records local review feedback.

The frozen v4 release remains review-only. Shared release v5 now binds the final
automatic policy: a complete valid uniquely resolved non-duplicate posted result
is eligible for Transactions; incomplete, invalid, ambiguous, duplicate,
abstained, interrupted, incompatible, or failed work is Review; valid `none`
settles without a transaction. Android adoption and runtime verification remain
in progress.

## Historical design selected 2026-09-01

PocketFinancer uses one shared deterministic analyzer, a high-recall triage policy,
one non-thinking Grounded Candidate Selector pass, strict host reconstruction, a
separate persistence gate, rich human labels, group-first corpus segregation, and
one local SQLite review workbench. The 2026-09-12 direct-extractor decision
superseded the candidate selector for new product operations; the design remains
here as historical rationale.

## Evidence

- Phase D measured no production selection; retain that conclusion.
- Candidate Protocol V1 improved extraction accuracy in its controlled comparison
  but failed its predeclared false-positive gate on every seed.
- Existing private splits omitted source rows and overlapped on IDs, exact bodies,
  normalized templates, and sender-template groups.
- The generated Semantic package covered incoming rows only, used a fixed
  timestamp/default currency, and silently downgraded invalid annotations.
- The reviewed 1,436-row package is overwhelmingly negative and therefore useful
  for negative safety/review, not as the primary production test.
- The first clean rebuild covers all 17,830 rows, produces zero protected-boundary
  overlap, and does not discard any of the 15 mapped legacy positives.
- Initial candidate coverage and triage counts show that deterministic support is
  incomplete. Retaining ambiguity is therefore safer and more honest than forcing
  binary truth.

## Alternatives considered

### Direct semantic generation (historical decision, superseded)

The 2026-09-01 design rejected this as the production route because it expanded
the validation surface. Evidence and contract work later showed that strict
source-span grounding and host-owned normalization could contain that risk while
avoiding deterministic candidate-coverage limits. The 2026-09-12 decision below
therefore supersedes this rejection for new operations.

### Byte-offset Candidate output

Rejected. UTF-8 byte counting is a host responsibility and is brittle for small
generative models and multilingual text. Exact spans stay in host candidates and
human truth.

### Candidate Protocol V1 as the final contract

Superseded, not erased. It proved candidate selection worth pursuing but lacked
the final none/abstain/review semantics, grounded direction requirement, currency
snapshot/provenance, and independent persistence/feedback contracts. Its measured
report remains historical evidence.

### Binary pre-filter

Rejected. A binary gate either wastes SLM work on obvious terminal messages or
silently loses ambiguous/missing-candidate cases. Tri-state storage disposition
plus selector action preserves safety and permits user assistance. Product work
should reduce `retain_review` through better analysis and feedback, not erase it.

### Body-wide OTP rejection

Rejected. Credential OTP is terminal only when no completed-event clause exists.
“Without OTP” and messages containing both posted and security clauses must remain
eligible for analysis/review.

### Regex segregation as truth

Rejected. Deterministic weak facets support browsing and sampling; only revisioned
human labels are truth. Ambiguous is a valid label.

### Random row splits after model iteration

Rejected. Template and sender repetition leaks across boundaries and biases later
choices. Whole normalized-template components are assigned before model work, with
the later-time holdout restricted to wholly new post-cutoff templates.

## Currency injection

The user's primary ISO-4217 currency is explicit configuration because sender/body
country inference is unreliable and setting changes must not rewrite history.
Explicit message currency overrides the snapshot. India-specific conventions live
in an extension profile.

## Transparency and feedback

The selected model pass is non-thinking. Product transparency therefore means
showing real deterministic evidence and real decoding/validation stages, not
displaying hidden or fabricated reasoning. `ProcessingTrace` records those stages;
`UserFeedbackEvent` binds future confirm/correct/reject actions to a trace and
canonical label revision.

## Safety and generalization implications

- All candidate values remain source-backed and operation-bound.
- Offline retention and runtime discard are separate.
- Recognition is broader than automatic persistence.
- Missing candidates and invalid labels produce review/error reasons.
- Private data, databases, source mappings, annotations, and derived rows remain
  ignored and local.
- Locale extensions can grow without baking Indian assumptions into the core.

## Repository and private-artifact disposition

| Disposition | Files or artifacts | Treatment |
| --- | --- | --- |
| Keep | `PRIVATE_DATA/all_sms.json` | Sole retained 17,830-row private source archive; ignored and never printed or committed. |
| Keep | The 1,436-row manually reviewed package, its source mapping, metadata, import report, key material, SQLite history, and backups | Human/negative-safety evidence and provenance; never regenerated or treated as the primary production test. |
| Keep | The 203-row fixture | Regression-only private evidence, not a protected product test. |
| Keep | Completed Phase D no-selection and Candidate Protocol V1 experiment reports, manifests, and compatibility code/config | Immutable historical evidence; measured results are not rewritten. |
| Adopt | `src/pocketfinancer_sms`, `configs/sms_processing`, the canonical private manifest, and SQLite storage/recovery primitives | The sole active analyzer, selector, label, corpus, and workbench path. |
| Supersede | Candidate V2 semantic byte-offset code/schema, Candidate Protocol V1, the former unified pre-filter, regex taxonomy, and active extraction-V2 plans/status/configs | Historical only. Active plans/configs were relocated under `docs/history` and `configs/history`; one compatibility symlink preserves the hash-sensitive old policy path. |
| Delete | Generated Semantic packages, overlapping random splits, old segregation outputs, duplicate extracted archives/ZIP, obsolete exploratory auto-label/build/export scripts and tests, `.DS_Store`, obsolete intermediate corpus runs, and empty workbench databases | Removed after source/member/hash and retained-copy checks. These derived outputs are not directly recoverable, but their raw source, human evidence, historical checkpoint branch, and current canonical run are retained. |

Review queues are views over the one canonical manifest. No deleted segregation is
allowed to re-enter the active path as human truth or as an independently generated
dataset.

## Unresolved risks

- The first analyzer is intentionally conservative; human review must measure
  false discard/invoke and candidate-oracle gaps by group.
- Counterparty and bare-amount enumeration need broader language/profile coverage.
- Multiple-event runtime support currently retains/abstains rather than persisting.
- Native implementation and migration verification are tracked by the native
  integration plan; the shared contract is frozen independently of rollout.
- Protected evaluation is not human gold until blind review/adjudication completes.
- No model has been trained or deployed on this foundation.

## 2026-09-06 — Freeze native integration release v1

Selected a hash-bound native release instead of allowing Kotlin and Swift ports to
infer behavior from prose. The release preserves stored `/1` compatibility and
adds `/2` analysis, result, trace, feedback, and selector-validation behavior.

Automatic persistence remains disabled. A native result can be recognized as a
posted event yet remain blocked by rollout, unresolved account identity, multiple
events, incomplete grounding, or any integrity failure. This separation is an
intentional safety boundary, not a temporary parser limitation.

Trace and feedback storage are append-only and operation-bound. User corrections
record grounding provenance and revisions but do not automatically become training
labels. Diagnostic transfer is encrypted and requires explicit consent.

## 2026-09-10 — Disable the selector wall-clock deadline

Native integration release v2 supersedes v1 for newly created operations. Selector
validation profile `/3` and processing configuration `/2` record
`parser_deadline_ms: 0`, meaning Android and iOS wait for the on-device selector to
finish instead of retaining a review after 60 seconds. Explicit user cancellation,
process interruption, durable claims, and heartbeat recovery remain active.

Release v1 and validation profile `/2` remain unchanged for historical operations.
Automatic persistence remains disabled; this release changes runtime completion,
not the rollout gate.

## 2026-09-12 - Adopt the direct SLM-primary extractor

New production-intended shared operations use one direct local extractor rather
than selecting deterministic candidate IDs. The model decides posted, none, or
abstain and, for posted, returns amount, currency, debit/credit direction,
account reference, optional counterparty, and exact Unicode-scalar evidence.

The deterministic analyzer remains in the runtime only as advisory context. Its
candidates and cues do not form an allowlist and cannot override the unchanged
message. The host retains ownership of strict JSON parsing, scalar grounding,
money normalization, account resolution, duplicate assessment, authoritative
receipt time, reason codes, review routing, and persistence.

This replaces the candidate selector because candidate coverage made model recall
depend on a deterministic extractor and prevented the SLM from correcting missed
or conflicting analysis. Direct semantic output expands validation surface, so
the choice is paired with grammar-constrained output, duplicate-key and
trailing-content rejection, exact source slicing, immutable configuration hashes,
sanitized Unicode goldens, and fail-closed review.

Candidate-selector contracts, prompts, profiles, goldens, and native releases
v1/v2 remain byte-for-byte frozen for stored-operation compatibility and
historical reproducibility. Future SFT labels use rich canonical-label/2 truth
and project directly to sms-extractor/1. No existing private label is silently
reinterpreted.

Native release v3 freezes the new shared boundary, but Android and iOS have not
integrated it. Automatic persistence remains disabled. This decision authorizes
neither model deployment nor data publication; protected human-gold evaluation,
runtime parity, privacy, license, and device review gates remain open.


## 2026-09-15 — Successor release v4 for system-managed model provenance

Evidence established that the frozen processing-config/3 requires a SHA-256 of a model file for every eligible runtime, while Apple Foundation Models exposes a system-managed model without an app-readable model artifact. A fabricated hash would falsely claim evidence that does not exist. Therefore native-integration-v4 and processing-config/4 are additive successors: eligible model_identity_kind=file_sha256 requires a real file SHA-256; an ineligible file-backed runtime may use null when no file can be read, while a supplied hash remains validated; model_identity_kind=system_managed_runtime always requires model_file_sha256=null and retains explicit model_identifier, runtime, OS, device, prompt, grammar, and validation provenance. Releases v1/v2/v3 remain frozen byte-for-byte. Automatic persistence remains disabled.

## 2026-09-21 — Make Review exception-only in a successor release

Product routing should not force a complete, strictly valid, uniquely resolved,
non-duplicate posted result through manual review. The target successor release
will insert that result into Transactions atomically. Incomplete, invalid,
ambiguous, abstained, interrupted, incompatible, and failed operations will retain
their source evidence and enter Review. A valid `none` decision remains a
separate terminal non-transaction outcome governed by the privacy policy.

This decision does not alter v4. V4 is frozen and `review_only`, so both apps
currently retain valid posted v4 results for review. Enabling the new route
requires an additive processing configuration/reason-code release, parity vectors,
migrations, recovery tests, native build/device evidence, protected quality
evaluation, and explicit rollout approval.

Review remains source-grounded: the full SMS, accessible per-field highlights,
and exactly one active native selection. Confirmation writes one atomic local
transaction; corrections append local revision-bound feedback and do not silently
become training labels.

Processing transparency must be real-time within each runtime's observable API.
Android exposes decoded token deltas. Apple Foundation Models exposes cumulative
structured-generation snapshots but not decoded token pieces or IDs, so iOS shows
those snapshots and explicitly labels token-level data unavailable. Neither app
may reconstruct or label hidden reasoning.

## 2026-09-22 — Freeze the final automatic-routing shared release

`native-integration-v5` implements the 2026-09-21 decision without changing any
v1-v4 byte or adding another routing engine. `processing-config/5` binds
`persistence-policy/2` in automatic mode. Complete valid uniquely resolved clear
posted results are eligible for Transactions, exceptions and duplicates retain
Review, and valid `none` creates neither a transaction nor a Review case.

`review-case/2` adds independently grounded field evidence with exact spans,
safe normalized values, validation state, originating stage, and explicit `slm`
versus `advisory_analyzer` origin. Analyzer evidence remains a suggestion and is
never represented as model output. Stored operations retain their original
release semantics; retry creates a distinct operation with explicit parent
lineage. The shared safety, Ruff, 820-test pytest, and diff gates pass locally;
Android, emulator, and physical-device verification are separate pending gates.
