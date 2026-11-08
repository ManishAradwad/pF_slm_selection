# Local SMS model-quality evaluation

This repository has one synthetic SMS semantic cohort and one deterministic host
scorer for two inference lanes. `evaluate-extractor` remains the Android-target
GGUF/llama.cpp host lane in WSL. `evaluate-apple-fm` is the macOS Apple Foundation
Models Python SDK lane. The suite declaration is
`configs/sms_processing/evaluations/model-quality-v1.json`; its fixture hash,
prompt/schema hashes, parser/evaluator hashes, runtime identity, locale, and
scoring versions bind each run. The synthetic labels live in the existing
`tests/sms_processing/fixtures/extractor/synthetic-suite.jsonl` fixture.

Both lanes feed the same extractor input (source plus advisory analyzer evidence)
to the model and use the same strict source parser and deterministic aggregate
scorer. The GGUF lane asks for nested direct-extractor JSON under its GBNF
grammar. The Apple lane uses a flat guided-generation schema because the
Python SDK accepts guided JSON; it maps every generated field to the direct
contract without repairing source text or offsets. The prompts and output
protocols differ. Reports mark the comparison non-causal and keep runtime facts
separate. WSL host latency is not Android device latency; macOS SDK latency is
not iPhone app latency.

## Stateful supplied-response mechanics runner

`pocketfinancer_sms.stateful_evaluation.SyntheticPipelineRun` adds the first
isolated end-to-end host mechanics lane. It accepts **invented messages only**
and a counted `SuppliedRuntime` (strings or simulated exceptions), not a live
model. `tests/sms_processing/test_stateful_evaluation.py` shows full sequences.

The implemented flow is reliable source admission/replay -> conservative
prefilter -> durable evidence/settings -> existing advisory analyzer and one
supplied response -> existing strict parsing/grounding/exact money -> existing
account/duplicate gates -> isolated ledger / no transaction / Review. A narrow
`confirm_account` helper simulates an explicit owner action on an otherwise valid
unknown-account Review; it atomically binds an alias and saves. It is not a model
success, annotation, or a replacement for native Review correction UI.

A fresh SQLite run refuses an existing file; resume requires identical settings,
analyzer configuration and host-source hashes. Use a separate path for each
comparison. Admission/create/claim/settlement use durable transactions. Reliable
replay reuses pending/busy/settled state before analysis or model work. Missing
identity never deduplicates text. Busy work is not stolen by opening another
instance. After the old runner has actually stopped, explicit
`recover_interrupted()` fences its unfinished attempts; replay then reports
retryable state and only explicit `retry_of` creates another operation. Completed
outcomes and parent operations are preserved. Supplied raw responses/attempts are
durable; they must never be mistaken for labels.

`mechanics_summary()` counts every admitted input (including excluded and replayed
inputs), operations, model calls, reused inputs, transactions and remaining Review.
No labelled scoring or quality denominator is claimed. Identity conflicts fail
closed without overwriting evidence. The existing exact-time payment fingerprint
is reused only as an interim mechanics gate; it does **not** settle different-alert
duplicate windows or prove payment-detection quality. Native ready-work recovery
may currently defer to visible Review rather than execute a stored snapshot.

This lane does not yet unify GGUF/harness or Apple's official Python SDK adapters,
match native request/settings end to end, provide full stage timings/scoring, or
accept private cohorts. Those are subsequent bounded slices. The live extraction
lanes below remain available unchanged; they are not the new stateful runner.
Product excluded/none retention is still an owner decision; synthetic run storage
is not permission to change native evidence retention. Dataset-backed quality is
pending owner annotation and cohort readiness. No local application suites are run
on the editing host; new tests require CI execution before claiming a pass.

## MacBook Air setup and first synthetic run

Use a compatible Apple Silicon Mac with macOS 26 or later, Xcode 26 or later
installed and its agreement accepted, and Apple Intelligence enabled with its
model downloaded. These are the [official SDK requirements](https://apple.github.io/python-apple-fm-sdk/).
For a fresh synthetic-only Mac checkout, run these exact commands in Terminal:

```bash
git clone --branch codex/sms-model-quality-eval --single-branch \
  https://github.com/pocket-financer/pocket-financer-intelligence.git \
  ~/pocket-financer-intelligence-sms-eval
cd ~/pocket-financer-intelligence-sms-eval
python3 --version
xcodebuild -version
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt -r requirements-apple-fm.txt
python -c 'import apple_fm_sdk; print("apple-fm-sdk import ready")'
umask 077
python scripts/run_sms_processing.py evaluate-apple-fm \
  --suite synthetic \
  --locale en_US \
  --output-dir PRIVATE_DATA/sms_processing/evaluations/apple-fm-synthetic-first
```

The CLI prints aggregate JSON only. Inspect the ignored local
`PRIVATE_DATA/sms_processing/evaluations/apple-fm-synthetic-first/report.json`
for the full aggregate report and `checkpoint.json` for private case results.
Never commit, paste, upload, or attach the checkpoint, model requests, prompts
containing SMS, snapshots, or predictions. The output directory is created with
mode 0700 and files with mode 0600. No hosted model, judge, or telemetry is used.

If the process is interrupted, rerun **the exact same command** with the same
checkout, SDK/runtime, locale, options, and output directory. Completed case IDs
are skipped. A changed fixture, code, prompt, schema, suite, environment, or
configuration fails closed; use a new output directory for a deliberate new run.
An unrecognized file in an output directory is rejected. A temporary file left
by an interrupted atomic checkpoint write is removed only after the durable
checkpoint remains the authority. Concurrent runs against one directory are rejected.

To confirm the live Mac uses the SDK, the report should say
`lane=apple_fm_python`, `provenance.runtime.simulated=false`, and
`provenance.runtime.model_identity_kind=system_managed_runtime`. The system model does not
expose a model-file SHA-256 or model revision through this lane. The Python SDK
currently exposes guided `respond(..., json_schema=...)` but its documented
`stream_response` is text-only, so cumulative **guided** snapshots are recorded
as unavailable. Decoded tokens, logits, confidence, and token throughput are
also unavailable. The SDK has an unsupported-locale error; it does not expose a
separate locale preflight in the Python API used here, which the report states.
See Apple's [session API](https://apple.github.io/python-apple-fm-sdk/api/session.html)
and [guided-generation documentation](https://apple.github.io/python-apple-fm-sdk/guided_generation.html).

## WSL GGUF host run on the same cohort

From the root of the native WSL `pocket-financer-intelligence` checkout on the
gaming PC:

```bash
source scripts/activate_wsl.sh
python scripts/run_sms_processing.py evaluate-extractor \
  --gguf /absolute/local/path/to/model.gguf \
  --suite synthetic \
  --output-dir PRIVATE_DATA/sms_processing/evaluations/gguf-synthetic-first
```

Use a real local GGUF path and a fresh output directory. Repeating the same
command resumes completed cases. The GGUF run records the real model-file hash,
llama-cpp-python version, grammar, embedded chat template requirement, decode
settings, and host timing. This is Android-target semantic quality on WSL, not
an Android phone measurement. Neither lane's synthetic scores support a product
quality or release claim.

## Current boundary

The Apple Python runner accepts only the synthetic cohort. Protected personal
SMS packaging, encrypted cross-device result transfer, exact Swift/iOS app
prompt and schema parity, iPhone inference, and Mac live results remain open.
Use the native app/device and protected human-gold gates in the evaluation
strategy before making product claims. Do not move private rows between WSL and
Mac to make this synthetic run work.
