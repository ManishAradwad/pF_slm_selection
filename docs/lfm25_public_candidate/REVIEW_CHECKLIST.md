# Manual review checklist

Candidate status: **unreleased; all rows pending**

Do not change `manual_review` or publish an artifact as part of this checklist. Record
review evidence in an approved review system outside the candidate rows.

- [ ] Confirm every row has only the documented schema fields.
- [ ] Confirm every `expected` value is literal `null` or has the four required keys.
- [ ] Inspect every transaction label against its synthetic message.
- [ ] Inspect every hard negative and confirm `expected` is literal JSON `null`.
- [ ] Check class, template-family, sender, debit/credit, amount, and counterparty
  coverage for the intended evaluation use.
- [ ] Review all rewrite and rejection counts in `audit_report.json`.
- [ ] Confirm every sensitive-data scan has zero blocked matches in accepted rows.
- [ ] Confirm safe-token findings are limited to generator-owned reserved formats.
- [ ] Investigate any changed n-gram threshold or near-duplicate threshold.
- [x] Aggregate-only model memorization probes are completed; their retained report
  contains no private text, private hashes, or generated completions.
- [ ] Review the reported aggregate verbatim-continuation and rare-n-gram match counts.
- [ ] Interpret the reported lower-loss membership AUC in light of the synthetic-train
  versus unseen-template-development distribution shift. Do not present it as an
  estimate of membership in the private SMS archive.
- [ ] Review possible trademark or institutional-confusion concerns.
- [ ] Review privacy and re-identification risk despite the synthetic provenance.
- [ ] Obtain a qualified data-rights and licensing review.
- [ ] Make an explicit release decision separately; none exists today.
- [ ] Make an explicit license decision separately; none exists today.
