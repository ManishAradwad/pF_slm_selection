# PocketFinancer Android Phase C baseline

Audit date: 2026-08-19
Baseline ID: `pocketfinancer-android-552ffbdf-phase-c`
Android committed baseline: `552ffbdfbd41773980aa249789b0cb508fdb19fd`
Locked production profile revision: `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`

## Scope and evidence boundary

This is the reproducible source and instrumentation baseline required before the
Extraction V2 Android prompt/output laboratory. It does not change the Android
application, the locked production profile, model selection, runtime defaults, or
either mobile worktree. It does not contain a model run.

The machine-readable baseline is
`configs/baselines/pocketfinancer-android-552ffbdf-phase-c.json`. Its verifier
hashes canonical Git blob bytes at the named commits. It never hashes the Android
working tree, reads app storage, or depends on local line endings. This matters
because the Android checkout had 149 unrelated dirty entries during the audit.
Those entries were neither inspected for baseline content nor changed, staged, or
committed.

The audit inspected committed application source and invented test material only.
It did not inspect SMS, sender, account, annotation, prediction, prompt/output,
model, or device-log data. iOS remained read-only.

## Production-profile relationship

The existing `pocketfinancer-android-current.json` profile remains byte-for-byte
unchanged. Its own SHA-256 is
`b818397851bb03fc3365710fe8ddfe82cadfe896525d931b93dbb8a6653c9d3f`,
and all 13 source hashes still verify against Android `a9b7df44`.

Android `552ffbdf` is a descendant three commits later:

1. `49b5cb0 feat: unify on-device SMS processing details`
2. `3e1e6f5 fix: show automatic SMS processing activity`
3. `552ffbd chore(deps): bump com.google.devtools.ksp ...`

Only one of the 13 profile-tracked paths changed:
`pipeline/src/main/java/com/pocketfinancer/pipeline/PipelineService.kt`. The change
adds invocation-scoped filter, inference, token-stream, cache/performance, and
persistence observer events. It also removes the sender from the legacy shared
pipeline-state message. The prompt assets, six-stage filter, parser, native
runtime, model-tier declaration, and locked production defaults did not change
within the profile-tracked set.

The profile is therefore reproducible evidence for `a9b7df44`, but it is not an
exact inventory of observability present at Android `552ffbdf`. Phase C records
both commits rather than silently advancing the production profile.

## Reproducible behavior capture

The baseline pins 31 production, schema, and test blobs that cover:

- the ordered six-stage deterministic filter;
- the outer system message, long user policy, seven demonstrations, grammar, and
  built-in-GGUF chat-template/fallback boundary;
- the defensive JSON parser and its accepted amount/type/account behavior;
- the custom Kotlin/JNI llama.cpp runtime and coordinator;
- context, batch, thread, CPU, KV-cache, sampler, thinking, answer, grammar, and
  prefix-session-cache settings;
- all five currently selectable Qwen/Gemma runtime variant identifiers;
- SMS receive/source timestamps, Room persistence, and current per-transaction
  runtime fields; and
- manual, historical, and automatic processing stages, live token views, runtime
  facts, terminal outcomes, and their focused tests.

Reproduce the immutable audit from the activated WSL environment:

```bash
python scripts/check_pocketfinancer_android_baseline.py \
  --android-repo /mnt/d/Personal_Projects/pocket-financer/pocket-financer-android
```

The verifier is allowed to report that the checkout is dirty. It verifies the
named commits, not uncommitted files. It fails on profile hash drift, missing Git
objects, ancestry/distance drift, a changed set of profile-tracked paths, or any
of the 31 baseline blob hashes.

## Current runtime and instrumentation

The committed Android runtime is CPU-only custom JNI on llama.cpp `b9198`.
`SlmModelSpec` supplies a 3,072-token context, zero GPU layers, automatic threads,
and model-specific thinking capability. JNI caps automatic threads at four, uses
`n_batch=512`, `n_ubatch=256`, flash attention, no mmap, and F16 KV cache when the
device advertises FP16 (otherwise Q8_0). Generation is greedy at temperature zero.
Thinking variants may use a 1,024-token thinking pass followed by a 256-token
answer; non-thinking variants use the direct 256-token answer path. GBNF is an
optional per-SMS snapshot and remains default-off.

The current scoped observer exposes these structural events:

1. deterministic filter started/rejected/passed;
2. inference started with model identity, thinking/grammar flags, and budgets;
3. runtime-emitted thinking and JSON token deltas;
4. inference completed with native performance and prefix-cache facts; and
5. persistence started after the cancellation boundary.

Native performance currently exposes prompt-evaluation milliseconds, generation
milliseconds, generated tokens, and a derived token rate. Prefix-cache facts expose
attempted/hit/prefix-token counts. The manual, historical, and automatic UI maps
these facts to pre-filter, cache/prompt, inference, persistence, and final
saved/already-saved/filtered/retry/error states.

These facts are useful runtime instrumentation, but they are not yet a complete
selection-grade aggregate:

- there is no content-free aggregate export in Android source;
- there is no end-to-end or per-stage latency clock;
- there is no peak RSS or battery measurement;
- native `t_load_ms` is read after the extraction operation resets llama.cpp
  performance counters and is not a verified model-load measure;
- per-transaction timing/model fields are row-linked private storage, not an
  authorized aggregate evaluation source; and
- source/static tests cannot establish JNI execution or phone performance.

## Persistence and source-time baseline gaps

The current ledger uses `Double` as the amount source of truth. That is not
Semantic V2 exact-money storage. `TransactionEntity.date` stores
`SmsMessage.date` as the SMS arrival timestamp. Source identity prefers
`date_sent` when positive and otherwise uses the received date, but the ledger
does not store an explicit timestamp-provenance enum. These are recorded baseline
gaps, not Phase C changes.

The current source handoff is idempotent and retains raw evidence in SQLCipher
backed queued-candidate or transaction storage. Phase C did not inspect that
storage. Exact-money migration, explicit timestamp provenance, Semantic V2
grounding, and any production adapter remain later-phase work after the required
selection and authorization gates.

## Host and Android-device evidence

Evidence classes are deliberately separate:

| Evidence class | Phase C result | What it proves |
| --- | --- | --- |
| Source/static | Captured and hash-verified | Exact committed declarations and tests at `552ffbdf` |
| Host | Lightweight Python verifier/tests only | Baseline, privacy, and aggregate-package logic; no runtime timing claim |
| Android device | Not measured | No device/emulator was attached; no inference was authorized |

The content-free ADB state probe returned `no devices/emulators found`. Phase C
did not start an emulator, install an APK, download a model, run inference, inspect
logcat, or substitute host timing for device timing.

`configs/contracts/pocketfinancer-android-runtime-evidence-v1.json` freezes the
aggregate-only package boundary for later authorized measurements. Host and
Android-device captures require different provenance, and device fingerprints
must be SHA-256 digests rather than raw identifiers. Only aggregate counts and
quantiles are accepted; raw messages, prompts, outputs, annotations, predictions,
and row-level records are structurally rejected.

Real aggregate packages remain ignored under `RESULTS/`. Validate one without
printing capture IDs or environment metadata:

```bash
python scripts/validate_pocketfinancer_android_runtime_evidence.py \
  RESULTS/path/to/aggregate-runtime-evidence.json
```

The exact committed invented fixture is available only behind an explicit smoke
flag:

```bash
python scripts/validate_pocketfinancer_android_runtime_evidence.py \
  tests/fixtures/pocketfinancer_android_runtime_evidence_synthetic.json \
  --invented-fixture
```

The invented host and Android-device numbers are schema smoke data only and are
not product evidence.

## Phase disposition

Phase C captures the reproducible current Android source baseline and records the
instrumentation and device gaps without selecting Direct V2, Candidate V2, a
model, a runtime profile, or a production default. No Phase D experiment was
started.

Phase D may begin in a new task from the final Phase C handoff commit. It may run
controlled, evaluation-only prompt/output comparisons for Qwen and Gemma only
within the existing privacy and authorization boundaries. Any host result must
remain host evidence, and missing Android-device evidence remains a gap rather
than an inferred pass.
