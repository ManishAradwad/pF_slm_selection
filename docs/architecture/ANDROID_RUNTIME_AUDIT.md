# PocketFinancer Android inference audit

Audit date: 2026-08-05
Android repository: `/home/tojinotzenin/pocket-financer-android`
Audited revision: `a6c8a11` (`main`)
Method: read-only source and history inspection; no Android files were changed

> **Superseded snapshot:** This document records Android `a6c8a11`. The current
> app profile is `a9b7df44`: the six-stage SMS filter is active, context is 3,072,
> thinking is model-dependent, and GBNF is optional/default-off. Non-thinking LFM
> models use direct 256-token generation. Use
> `configs/contracts/pocketfinancer-android-current.json` and the unified pipeline
> for new work; retain the remainder below only as historical audit evidence.

## Verdict

The selection repository matches the Android prompt assets and chat roles. It does
not yet match the app's current preprocessing, generation, parser, model-selection,
or native runtime. The app also has integration/runtime blockers that make a target
device score premature.

## Current path

The manifest registers `SmsReceiver`, which sends received messages to a shared
repository channel. No app component currently collects that channel and calls
`PipelineService.enqueue()`. The reachable UI path is the synthetic "RUN TEST SMS"
debug flow.

If a message is queued manually, `PipelineService.processSingle()` builds a prompt
immediately. There is no sender, currency, account, transaction-verb, OTP, collect,
or mandate prefilter in the current Android source or history.

Key sources:

- `app/src/main/AndroidManifest.xml:26`
- `sms/src/main/java/com/pocketfinancer/sms/SmsReceiver.kt:20`
- `pipeline/src/main/java/com/pocketfinancer/pipeline/PipelineService.kt:137`
- `app/src/main/java/com/pocketfinancer/ui/settings/SettingsViewModel.kt:206`

## Exact parity that does exist

`PromptBuilder` loads the long extraction instruction and seven demonstrations,
then appends the current sender/SMS under `### YOUR TASK`. `PipelineService` sends:

- system: `You are a helpful financial SMS extraction assistant.`
- user: the long policy, seven examples, and current SMS

This matches `lfm25.android_contract.android_extraction_messages`.

| Asset | SHA-256 in both repositories |
|---|---|
| Long system policy | `16e042a07a18165e1cd0b1c0d0cd3bcee67f64825df8adc74e568b3eadffd64a` |
| Seven examples | `ea4e57c646f2232b5e1d24c1211b8ee6ac68cfef2f69c52ac9ae3765116749aa` |
| GBNF grammar | `c321daca16ea3dbdf4269c6504f7cbab5e587d1ce849e3b79133e5449d1c7939` |

Sources: `pipeline/.../PromptBuilder.kt:32-74` and
`pipeline/.../PipelineService.kt:115-157`.

## Native runtime and decode

Android does not use `llama.rn`. It has a custom Kotlin `LlamaEngine` and JNI bridge
that FetchContent-builds llama.cpp tag `b9198`. The current build is CPU-only,
uses at most four threads, zero GPU layers, flash attention, Q8 KV cache,
`n_batch=512`, `n_ubatch=256`, and disables memory mapping.

Every model is forced through two phases:

1. Append `<think>\n`; greedily generate up to 1,024 tokens without grammar and
   stop on `</think>`.
2. Append the captured thought plus `</think>\n`; greedily generate up to 256
   answer tokens with `sms_extraction.gbnf` and no stop string.

There is no repetition-penalty sampler and no grammar toggle. The path ignores the
model tier's `hasThinkingMode`, so non-thinking models are also forced through it.

Sources: `inference/.../LlamaEngine.kt:183-219`,
`pipeline/.../PipelineService.kt:151-157`, `inference/.../llama_jni.cpp`, and
`inference/src/main/cpp/CMakeLists.txt:8-29`.

## Context incompatibility

The app loads every tier with `n_ctx=1024`. Across the locked 203 rows, the exact
prompt is 1,862-2,017 LFM2.5-350M tokens before any generated thought or answer.
Zero rows fit. A faithful two-pass path needs enough context for prompt + thought +
answer; approximately 4,096 is the sensible first test, then it should be measured
per chosen tokenizer and maximum SMS length.

The JNI overflow branch currently discards all but the final 64 input tokens. That
removes nearly all instructions and examples and can remove most of the current SMS.

Sources: `app/.../SettingsViewModel.kt:165-169` and
`inference/.../llama_jni.cpp:151`.

## Model selection

Android defines Qwen3-1.7B Q8/Q4, Gemma 4 E2B Q8/Q4, and Qwen3-0.6B Q8 tiers. It
defines no LFM2.5-350M model, download, memory profile, or generation path. Current
RAM branches select Qwen for every viable tier; Gemma branches are shadowed by an
earlier `ram >= 3.5` condition. The settings UI displays the automatic choice and
does not provide a model picker.

Source: `hardware/.../SlmSelector.kt:28-165`.

## Parser differences

Android strips thought blocks, accepts case-insensitive `null`, extracts from the
first `{` through the last `}`, coerces numeric amount strings, requires amount > 0,
accepts only debit/credit, and permits a missing counterparty. Invalid output becomes
`SKIPPED`, the same state as a valid non-transaction. A blank counterparty is stored
as `Unknown Merchant`.

The selection scorer instead requires exactly four fields, rejects extras, handles
amount and counterparty normalization differently, and extracts the first balanced
JSON object. These must be unified or reported as separate metrics.

Sources: `pipeline/.../ExtractionParser.kt:37-91` and
`pipeline/.../PipelineService.kt:178-203`.

## Native blockers observed in source

These require an Android build/runtime task before trusting device inference:

- the JNI chat-template function appears to redeclare `msgs` in one scope;
- the chat-template output buffer is fixed at 4,096 bytes with no resize/retry;
- message counting treats every JSON `{` in few-shot examples as a message;
- the grammar sampler branch has no explicit greedy/distribution selector;
- stop handling can duplicate `</think>`;
- KV state is not explicitly cleared between phases/messages; and
- no native GGUF/JNI generation test exists.

This was a source audit, not a successful native build, so each item should be
confirmed in the Android repair task.

## Contract decision before the next model run

Choose one contract and implement it in both repositories:

1. **Direct LFM contract:** use the model's built-in chat template, single-pass
   answer generation, a measured context of at least 3,072, and optionally add the
   six-stage prefilter to Android after validating it. This best matches the direct
   adapter we already trained.
2. **Current Android two-pass contract:** train/evaluate prompt + 1,024 thought +
   grammar answer with enough context. This spends much more latency and is a poor
   default for a 350M non-agentic extraction model.
3. **Candidate contract:** add the same deterministic candidate extractor, compact
   selector schema, strict parser, and source reconstructor to Android. This is the
   strongest clean 350M research path but requires app architecture work.

For LFM2.5-350M, option 1 is the lowest-risk direct integration; option 3 is the
most promising quality architecture. Whichever is chosen needs cross-repository
golden tests for prompt hashes, grammar hash, context, decoding, and parser behavior.
