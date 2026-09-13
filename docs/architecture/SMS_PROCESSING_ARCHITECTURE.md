# SMS Processing Architecture

Status: **active architecture and implementation authority**
Executable package: `src/pocketfinancer_sms`
Private corpus configuration: `configs/sms_processing/archive-india-inr.json`

## Active v3 outcome

The production-intended shared foundation is SLM-primary. Android and iOS have
not integrated it. The frozen shared contracts and Python implementation are the
authority and parity oracle for future native ports.

For each incoming message:

1. The coordinator creates an immutable processing-config/3 snapshot with
   operation identity, platform receipt time, primary currency, runtime hashes,
   and automatic persistence disabled.
2. The deterministic analyzer produces advisory candidates and cues. Advisory
   evidence is visible to the extractor but is never an answer allowlist.
3. The host builds sms-extractor-input/1 from the unchanged message and makes one
   greedy, non-thinking local GGUF attempt.
4. The extractor returns only none, abstain, or one posted extraction containing
   semantic values and exact Unicode-scalar source evidence.
5. The host strictly parses and grounds the output, normalizes exact money, and
   resolves the account reference against existing accounts.
6. The coordinator performs duplicate assessment and the typed persistence gate.
   Recognition and persistence remain separate.
7. Invalid, interrupted, abstained, unresolved, ambiguous, or blocked operations
   become revisioned local review cases. Receipt time is read-only.

```text
platform message + immutable configuration
                 |
                 +--> advisory deterministic analysis
                 |
                 v
        one direct local SLM extraction
             /          |          \
          none       abstain       posted values + scalar spans
             \          |          /
                 strict host validation
                          |
               normalization + account resolution
                          |
               duplicate + persistence gates
                    /             \
             typed result      local review
```

### Active executable boundaries

| Concern | Authority |
|---|---|
| Advisory analysis and source preservation | `analyzer.py`, `structural_text.py`, `types.py` |
| Extractor input, parsing, grounding, normalization | `extractor.py` |
| Runtime/configuration eligibility | `configuration_v3.py` |
| Account resolution | `account_resolution.py` |
| Operation coordination and typed result | `processing_v3.py` |
| V3 persistence safety | `processing_v3.py` |
| Rich human truth and direct target projection | `labels.py` |
| Trace, review, and feedback | `trace.py`, `feedback.py` |
| Offline GGUF evaluation | `evaluation.py` |

The model cannot set receipt time, reason codes, operation identity, normalized
account identity, duplicate status, or persistence. No taxonomy regex or second
classifier model independently decides semantic truth.

## Historical candidate-selector architecture (native v1/v2)

The following flow remains documented to reproduce stored candidate-selector
operations and native releases v1/v2. It is not the model contract for newly
created shared operations. Historical assets remain byte-for-byte frozen.

For one incoming message, the intended runtime sequence is:

1. The host snapshots operation configuration: primary ISO-4217 currency, enabled
   locale profiles, message timestamp and provenance, direction metadata, and an
   operation identifier.
2. `DeterministicSmsAnalyzer` preserves the source text and enumerates clauses,
   exact host spans, money, direction, account, counterparty, state cues, and
   aggregate-safe reasons.
3. `evaluate_triage` applies policy to that same analysis. It returns both a
   storage disposition (`invoke`, `discard`, or `retain_review`) and an orthogonal
   selector action (`run_normal`, `run_assistive`, or `skip`).
4. When the selector action permits it, the host packages the message and its
   source-backed candidates for one greedy, non-thinking SLM pass.
5. The SLM emits only `none`, `abstain`, or one `posted` object selecting four
   candidate IDs. It never emits offsets or canonical values.
6. The host strictly validates every ID and reconstructs a rich semantic result.
7. The persistence gate independently decides whether the reconstructed result is
   safe to save automatically. Recognition does not imply persistence.
8. Anything unresolved is retained for local/user review. A processing trace can
   show deterministic analysis, the compact model stream and raw output,
   validation, reconstruction, and persistence decision without inventing chain
   of thought.

```text
platform message + configuration snapshot
                 │
                 ▼
      deterministic structural analyzer
                 │
                 ├── terminal certainty ───────────────► discard (offline row retained)
                 │
                 ├── incomplete/ambiguous ─────────────► retain_review
                 │                                      ├── run_assistive when grounded
                 │                                      └── skip when not representable
                 │
                 └── one grounded event candidate ─────► invoke / one SLM pass
                                                          │
                                    none ◄────────────────┼──────────────► abstain
                                                          │
                                                          ▼
                                               posted candidate IDs
                                                          │
                                             strict host reconstruction
                                                          │
                                  ┌───────────────────────┴──────────────────────┐
                                  ▼                                              ▼
                           auto persistence                               retained review
                         (all safety gates)                           (any unresolved gate)
```

### Historical executable boundaries

| Concern | Authority |
|---|---|
| Analyzer and host evidence | `analyzer.py`, `structural_text.py`, `types.py` |
| Currency snapshot and exact money | `currency.py`, `profiles.py` |
| Pre-filter policy | `triage.py` |
| Compact model contract and reconstruction | `selector.py` |
| Automatic persistence | `persistence.py` |
| Human truth and target projection | `labels.py` |
| Observability and user feedback | `trace.py`, `feedback.py` |
| Canonical private corpus and pools | `corpus/` |
| Local review UI and durable state | `workbench/` |

No taxonomy regex independently decides truth, and no second classifier model is
part of the production path.

### Historical analyzer output

The versioned analysis object contains:

- a message- and configuration-bound `analysis_id`;
- an NFKC/casefold/whitespace structural view used only for matching, with a
  fingerprint recorded while all returned evidence remains unchanged source text;
- unchanged source-backed clause spans in character and UTF-8 coordinates;
- amount candidates with exact minor units, currency, and provenance;
- completed direction evidence candidates;
- account/card/VPA and counterparty candidates;
- explicit absent account and counterparty candidates;
- failure, negation, pending, due, request, balance, promotional,
  administrative, credential, and payment-rail cues;
- source-safe reason codes and input/direction metadata.

UTF-8 offsets are host metadata and human truth. They are never model output.

### Historical state transitions

```text
received
  → analyzed
  → discarded_terminal | retained_review | selector_pending
selector_pending
  → selector_none | selector_abstain | selector_posted | selector_invalid
selector_posted
  → reconstructed | reconstruction_failed
reconstructed
  → persisted_automatically | retained_review
retained_review
  → draft → submitted → revealed (protected only, explicit) → revised/adjudicated
```

Malformed, unknown, ambiguous, inconsistent, unsupported, cross-message, or
cross-clause core selections fail closed to review. They never become negative
truth and never silently persist.

## Historical measured candidate baseline

The current configured private rebuild represents all 17,830 archive rows exactly
once. It is intentionally conservative: 1,453 rows currently receive normal
selector invocation, 125 may receive assistive invocation, and 16,252 skip the
model. This is a starting diagnostic, not a claim that
the deterministic analyzer is complete. Human segregation work will measure and
improve those boundaries without using weak outputs as ground truth.

## Frozen native release v3

`configs/sms_processing/contracts/releases/native-integration-v3.json` binds the
direct extractor schemas, profile, prompt, grammar, reason registry, algorithms,
and sanitized goldens by SHA-256. Native ports reproduce those bytes and
decisions rather than reinterpret prose. Releases v1/v2 and every selector asset
remain immutable historical compatibility. V3 adds direct semantic extraction,
Unicode-scalar grounding, versioned review cases, explicit account resolution,
hash-chained traces, revision-bound feedback, and rich canonical-label/2 truth.

Automatic persistence is deliberately disabled. Android and iOS remain
unintegrated until their review-only ports reproduce the frozen hashes, scalar
conversion rules, and golden cases.
