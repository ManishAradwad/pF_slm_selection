# PocketFinancer Android protocol laboratory — Phase D audit

Status: completed and reviewed on 2026-08-22, with no protocol, model,
runtime variant, or profile selected.

This report supersedes the Phase C handoff as the current Android experiment
record. Phase C evidence remains frozen and unchanged. Phase E did not start.

## Outcome

Phase D implemented ten controlled, evaluation-only profiles: Direct V2 and
Candidate V2 independently for each of Qwen3-0.6B Q8, Qwen3-1.7B Q4/Q8, and
Gemma 4 E2B Q4/Q8. Every profile remains unselected. The four-row invented
smoke fixture is too small for a decision, no reproducible quality baseline or
device budget exists, protected/blinded scoring was not authorized, and no
Direct/Candidate Android protocol harness exists. Every overall gate therefore
failed closed with `decision=no_selection`.

The committed fixture uses the Workbench V2 schema label `protected_test` to
exercise split-lock behavior. It is public invented synthetic material, was not
blind, and is not a protected/private evaluation. The result explicitly retains
`protected_scoring_not_authorized` as a gap.

## Immutable bindings and implementation

- SLM branch: `codex/extraction-v2-phase-d`.
- Phase C starting handoff:
  `61672935d8af21179a40346e097610c2ba09da46`.
- Protocol-lab implementation:
  `51112815f854e00a933db9bc9df18385a4923c94`.
- Repository-safety follow-up:
  `3c77cd60710649fb074eea73cba36e0dbf59d0ab`.
- Strict-lint provenance correction:
  `adc7b0ff795d7137a8874217d29a4590fcd7ec7b`.
- Versioned CUDA host runtime:
  `965620d09136c7fc689da5bb7d974db80e6f9db3`.
- Android authority:
  `552ffbdfbd41773980aa249789b0cb508fdb19fd`.
- Locked production-profile revision:
  `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`.
- Phase C baseline manifest SHA-256:
  `9274e5a63524b46bb4149e11d5190bae4ebcfef15170a69af77ad050b31167fc`.
- iOS read-only authority:
  `04f770b235f860080ecd96adad1a5d011f3c2c2c`.

The Android and iOS native Windows Git worktrees were clean before and after the
authorized build/device work. Neither mobile source tree, production default,
locked production profile, deployment behavior, or release state changed.

## Frozen Phase D artifacts

- Experiment manifest file:
  `c38214b536d14b04c4322c790f79d7c0d36743d125b0aa60b39c1d4d5f401c8a`.
- Resolved lab-manifest binding:
  `6368f6d3b8861958e4f5164b047b52761490cd40b5c7e47c59eaf23d727297f0`.
- Lab contract:
  `f0af91d3eaed9e34ec24411eb344dbd8e79831710f6d25f29024c1f4c9075131`.
- CUDA host/runtime profile:
  `1c68fe057f20c957b4f3a2bf79ce798b3dfc38a0463ac73903f9d9914a096ab0`.
- Parser/lab engine:
  `9615142da913044a2c819fb7801a99c47ba29582db0fa3c4f2cb7e2aab72db85`.
- Runner:
  `e67f8818d88325ace6e55cf49d92c77457cca6873bebe76ff90a121bf3b70e20`.
- Guide:
  `155da8b3b2a8ae3d16aa2e52cb9512d2c4976dfc0f283ab8542ff5b49564915b`.
- Qwen Direct/Candidate prompts:
  `66c7e616cdea7346e1250eb4b41de43a25b5911d28aae4ebd34f373381dc1cae` /
  `440720a9f8f2bcb8b5cbac0d7baeed6f0ce9172f0a4b4bdcdbc69a207170f4c2`.
- Gemma Direct/Candidate prompts:
  `2feb51c67eb7a90f627fe4fbfa8738e3f3231f5a182728124e8bcca8d95c2853` /
  `72c15f69f0e645c4478ce7957823490bbb741344766cb3528676070c14a1cc83`.

The runner verified all five already-local GGUF artifact hashes before inference.
No model was downloaded, trained, fine-tuned, merged, converted, or quantized.

## Evidence classes

### Source/static

Both adapters passed all 13 invented Semantic V2 conformance vectors with zero
uncaught exceptions. Prompt family, chat-template strategy, parser, model
artifact, decode/runtime settings, evaluator, fixture, and baseline provenance
are bound independently for every profile. Results are never pooled across
runtime variants or model families.

### Host GGUF

The ignored aggregate-only result is
`RESULTS/phase_d/pocketfinancer-android-phase-d-synthetic-cuda-v1.json`,
SHA-256
`933c2e6aafac0209bda2c56455af81c97eb1806ed4868e3182f2286da8ad0828`.
It records ten measured `host_gguf` profiles, no row predictions or raw model
output, and implementation commit
`965620d09136c7fc689da5bb7d974db80e6f9db3`.

The controlled run used the CUDA-linked `llama-cpp-python==0.3.20` runtime on
NVIDIA device 0 with required all-layer offload (`n_gpu_layers=-1`), 3,072
context tokens, temperature zero, seed 17, and each GGUF's verified embedded
chat template. The binding reported CUDA offload support and the runner logged
an RTX 4070 at compute capability 8.9. Live `nvidia-smi` sampling during the
evaluator recorded 67–76% SM utilization, 54–60% memory-controller utilization,
2,870 MiB VRAM, and approximately 119–122 W. A transient unrelated load observed
during preflight was allowed to fall to idle before launch. This is observed
NVIDIA GPU execution, but its latency remains desktop-host timing—not emulator,
simulator, or physical-phone timing.

| Runtime variant | Protocol | Scope accuracy | Invalid output rate | Host p95 ms | Nonzero parser failures |
|---|---:|---:|---:|---:|---|
| Gemma 4 E2B Q4 | Candidate V2 | 0.25 | 0.75 | 979 | protocol shape 3 |
| Gemma 4 E2B Q4 | Direct V2 | 0.00 | 1.00 | 1,969 | non-JSON 4 |
| Gemma 4 E2B Q8 | Candidate V2 | 0.25 | 0.75 | 1,281 | protocol shape 3 |
| Gemma 4 E2B Q8 | Direct V2 | 0.00 | 1.00 | 1,851 | non-JSON 1; semantic validation 3 |
| Qwen3 0.6B Q8 | Candidate V2 | 0.00 | 1.00 | 1,777 | protocol shape 4 |
| Qwen3 0.6B Q8 | Direct V2 | 0.00 | 1.00 | 1,884 | semantic validation 4 |
| Qwen3 1.7B Q4 | Candidate V2 | 0.00 | 1.00 | 3,795 | non-JSON 1; protocol shape 2; semantic validation 1 |
| Qwen3 1.7B Q4 | Direct V2 | 0.00 | 1.00 | 5,854 | semantic validation 3; unclosed thinking 1 |
| Qwen3 1.7B Q8 | Candidate V2 | 0.00 | 0.75 | 7,761 | unclosed thinking 3 |
| Qwen3 1.7B Q8 | Direct V2 | 0.00 | 1.00 | 7,792 | semantic validation 1; unclosed thinking 3 |

These four-row estimates are diagnostic only. Apparent differences do not
select Candidate V2 or establish family portability. All sample, threshold,
baseline, operational-budget, memory, battery, recovery, availability, and
device-protocol gates remain failed or missing. Safety/provenance subchecks that
passed do not turn the overall gate into a pass.

The original CPU-only aggregate remains preserved at
`RESULTS/phase_d/pocketfinancer-android-phase-d-synthetic-v1.json` as
historical evidence and was not overwritten or relabelled. Its older parser,
manifest, implementation, and runtime hashes do not validate as the current
CUDA result.

Standing user authorization now requires best-available applicable hardware.
GPU-suitable work may use NVIDIA CUDA; Windows Android emulators and macOS
Android emulators/iOS simulators may be used when available. No macOS execution
host was attached to this task, so no macOS emulator/simulator result is
claimed. Host CPU, host GPU, each emulator/simulator, and physical devices remain
separate evidence classes.

Future shared-GPU work requires prior notice naming the workload and reason. It
must be deferred while the user reports an interactive GPU workload.

### Android emulator smoke

After explicit device authorization, an API 35 x86_64 Android emulator was used
for baseline runtime smoke only. The debug APK was force-rebuilt from the
native-clean Android checkout at the bound commit with
`:app:assembleDebug --rerun-tasks --no-daemon`. The build passed and produced
`com.pocketfinancer.debug` version `1.1.0-debug`, supporting arm64-v8a and
x86_64, with APK SHA-256
`a88b26d495b695f9e2af52ba45415033e1772a94768f1b15c9a0007a5b721b39`.

The pre-existing emulator install initially reported SMS permissions granted.
No message content, app storage, or device log was inspected. The app was
force-stopped and `READ_SMS`/`RECEIVE_SMS` were revoked before the qualifying
content-free cold launch. That launch returned `Status: ok`, resumed
`MainActivity`, and kept a live app process. Its 2,328 ms launch time is
emulator launch evidence only, not inference or phone timing.

Reviewed invented-only instrumentation then passed:

- `:app:connectedDebugAndroidTest`: 13/13 tests.
- `:inference:connectedDebugAndroidTest`: 1/1 test.

A final post-instrumentation audit found both SMS permissions granted again. No
message content, app storage, or device log was inspected. The app was
force-stopped, both permissions were revoked again, and a final content-free cold
launch returned `Status: ok` with an 8,162 ms emulator-only launch time.
`dumpsys package` then confirmed `READ_SMS` and `RECEIVE_SMS` both
`granted=false`, and the app remained running. The earlier 2,328 ms and final
8,162 ms observations are emulator launch evidence only; neither is model
inference or physical-phone timing.

No Direct V2 or Candidate V2 harness exists in the Android application. Zero
Android protocol profiles were measured, no model inference ran on the emulator,
and this smoke does not satisfy a protocol-comparison, physical-phone, memory,
battery, or recovery gate.

### HF host and physical phone

No HF-host evidence was produced. No physical Android phone was attached or
measured. Emulator evidence is never described as phone evidence.

## Privacy and selection

Only committed invented synthetic fixtures were used for model-facing work. No
private SMS, sender, account, annotation, row-level prediction, stable mapping,
app storage, log content, or private generated artifact was inspected, printed,
copied, committed, or transmitted. Raw prompts/model outputs remained in memory
and were not retained.

All `selected_profile_id` values remain null. Direct V2, Candidate V2, every
model, every runtime variant, and every profile remain unselected. Production
defaults and deployment state are unchanged.

## Verification

The final Phase D gate was run from the activated native WSL environment:

- Entry and final program-state hash lock: 1 passed.
- Phase D target plus program-state test: 13 passed.
- Repository-safety unit tests: 25 passed.
- Repository safety: passed, with only the pre-existing publication-review
  exception for `DATA/extraction_ds.jsonl`.
- `ruff check .`: passed.
- `ruff check --select E4,E7,E9,F lfm25 scripts tests`: passed.
- Full `pytest -q`: 603 passed.
- `git diff --check`: passed.
- CUDA capability: `llama_supports_gpu_offload=True`; local llama.cpp linked
  to CUDA 12, cuBLAS, and `libggml-cuda`.
- Live GPU execution: RTX 4070 sampled at 67–76% SM, 2,870 MiB VRAM, and
  approximately 119–122 W while the evaluator ran.
- Immutable Android baseline verifier: 31 baseline blobs and 13 profile blobs
  verified; Android HEAD matched `552ffbdf`.
- Android forced debug build: passed.
- Android emulator instrumentation: 14 passed.
- Native Windows Git status: Android and iOS clean at their expected commits.

The additional pipeline check preserved its expected provenance refusal: the
auto-discovered WSL Android clone is at `a6c8a11`, not the locked production
profile revision `a9b7df44`. The explicitly named Android baseline verifier
passed. No guard was weakened and no production pipeline/profile/lock changed.

## Remaining gaps and next boundary

Phase D is complete with a valid no-selection outcome, not with a passing
candidate. Protected/blinded evaluation, user-approved real labels and splits,
device budgets, reproducible baselines, Android protocol inference, physical
phone latency/memory/battery/recovery, and every production integration decision
remain absent.

The next allowed phase is E, the LFM2.5-350M data, fine-tuning, and quantization
program. It must start in a separate task with an explicit experiment design and
any required authorization for real labels/splits, training, conversion, or
downloads. The standing hardware policy already authorizes CUDA when an
otherwise-authorized workload benefits from it; it does not authorize the
workload itself. Phase E has not started here, and Phase D evidence does not
establish Android deployability.
