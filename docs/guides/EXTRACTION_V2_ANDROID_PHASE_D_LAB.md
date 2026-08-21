# Extraction V2 Android Phase D protocol laboratory

This laboratory compares two evaluation-only output hypotheses independently for
each Android Qwen/Gemma runtime variant. It does not change the Android app,
production prompts, the locked production profile, model selection, deployment,
or release state.

## Frozen boundary

Every profile binds Android
`552ffbdfbd41773980aa249789b0cb508fdb19fd` and Phase C baseline manifest
SHA-256
`9274e5a63524b46bb4149e11d5190bae4ebcfef15170a69af77ad050b31167fc`.
The experiment manifest is
`configs/experiments/pocketfinancer-android-phase-d-v1.json`. It declares ten
profiles: Direct V2 and Candidate V2 for each of Qwen3-0.6B Q8,
Qwen3-1.7B Q4/Q8, and Gemma 4 E2B Q4/Q8. Results are never pooled across a
runtime variant, and neither protocol is presumed portable between Qwen and
Gemma.

Direct V2 asks the model to emit the Semantic V2 core, including grounded values
and exact minor units. Candidate V2 asks only for scope/status/cardinality and
UTF-8 evidence spans; the host derives amount, currency, minor units, direction,
account, and counterparty values with the frozen Semantic V2 reference. Both
adapters reject model-supplied source timestamps, unknown fields, malformed JSON,
and ungrounded evidence. The trusted SMS timestamp is injected after parsing.

## Evidence classes

- `source_static` covers prompt/adapter validation and all 13 invented Semantic
  V2 conformance vectors.
- `host_hf` is not part of this laboratory and cannot be inferred from GGUF.
- `host_gguf` uses already-local SHA-256-verified artifacts and the versioned
  CUDA `llama-cpp-python` profile with required all-layer offload, plus the
  committed four-row invented Workbench smoke fixture. Host GPU latency is not
  emulator, simulator, or physical-phone latency.
- `android_device` is recorded independently. A baseline app/runtime smoke does
  not become protocol-comparison evidence unless an evaluation-only Direct or
  Candidate harness runs the exact profile on the device.

The frozen Phase B threshold policy still applies. Four invented smoke rows are
far below its protected sample sizes, are not blinded/protected scoring, and
cannot select a protocol. Missing baseline, operational-budget, memory, battery,
recovery, or device-profile evidence is a failed gate, never a pass.

## Safe commands

From the activated native WSL environment, validate the manifest and adapters
without loading models:

```bash
python scripts/run_pocketfinancer_android_phase_d_lab.py
```

Hash all five expected local artifacts without downloading anything:

```bash
python scripts/run_pocketfinancer_android_phase_d_lab.py --verify-artifacts
```

After the implementation commit exists, confirm the CUDA binding reports GPU
offload support, then run the controlled CUDA GGUF smoke and write only aggregate
evidence under the ignored `RESULTS/` root:

```bash
python -c 'from llama_cpp import llama_cpp; assert llama_cpp.llama_supports_gpu_offload()'

python scripts/run_pocketfinancer_android_phase_d_lab.py \
  --run-host-gguf \
  --implementation-commit FULL_IMPLEMENTATION_COMMIT \
  --device-smoke-json RESULTS/phase_d/pocketfinancer-android-phase-d-emulator-smoke-v1.json \
  --output RESULTS/phase_d/pocketfinancer-android-phase-d-synthetic-cuda-v1.json
```

Sample `nvidia-smi` while the runner is active and record the NVIDIA device,
utilization, VRAM, and power observation in the reviewed aggregate handoff.

The runner keeps prompts, raw output, and row predictions in memory only. The
committed result, if reviewed and copied from `RESULTS/`, may contain aggregate
metrics, counts, failure taxonomy, and hashes only. It must retain
`selected_profile_id: null`, `decision: no_selection`, and
`phase_e_started: false`.
