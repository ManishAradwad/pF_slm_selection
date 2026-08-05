## Summary

<!-- What engineering or research outcome does this PR deliver? -->

## Scope and end-to-end behavior

<!-- Identify affected contract, data, training, evaluation, conversion, or runtime
paths. State what is deliberately unchanged. -->

## Experiment and app alignment

<!-- Note the Android revision/profile, dataset roles, model/quantization, metric
interpretation, and runtime-parity boundary when relevant. Write "Not applicable"
for a change that cannot affect them. -->

## Verification

<!-- List exact commands and results. Distinguish lightweight, CUDA, GGUF, and
target-device checks. -->

- [ ] Repository safety check passes
- [ ] Relevant automated tests pass
- [ ] Ruff/static checks pass
- [ ] Pipeline/profile check passes, or is not applicable
- [ ] Runtime/device verification is complete, or its gap is documented

## Privacy, reproducibility, and compatibility

<!-- Cover private data, leakage, seeds/hashes/provenance, output compatibility,
memory/latency, cache/artifact reuse, and rollback implications. -->

## Evidence

<!-- Aggregate metrics, non-sensitive logs, screenshots, or links to reports. Never
paste raw SMS, identifiers, private per-row output, credentials, or weights. -->

## Checklist

- [ ] I read `AGENTS.md` and verified relevant prose against executable code
- [ ] This change is focused and does not include generated/private artifacts
- [ ] The PR title uses a Conventional Commit type
- [ ] The 203-row regression fixture is not presented as a fresh test
- [ ] HF, host GGUF, and Android-device evidence are labeled truthfully
- [ ] No model, dataset, or application publication is implied by this PR
