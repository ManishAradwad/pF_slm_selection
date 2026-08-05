# License and data-rights review record

Status: **open; no legal conclusion**

## Engineering facts recorded for review

- The generator uses deterministic templates and curated fictional token sets.
- Private messages are not generation inputs.
- A local private corpus may be read in memory only for similarity rejection.
- The auditor persists aggregate counts and scores, not private text, hashes,
  identifiers, mappings, or candidate-to-private linkage.
- The private source archive is not mutated or copied by either script.
- Candidate rows carry fresh public IDs and remain `manual_review=pending`.
- Generated artifacts remain in the Git-ignored `PUBLIC_CANDIDATE/lfm25/` tree.

## Questions requiring qualified review

- Does the proposed use create trademark or false-affiliation concerns?
- Is the documented synthetic process and audit evidence sufficient for the intended
  jurisdictions and distribution method?
- Are contractual or database-right restrictions implicated by the private corpus
  being used solely for local similarity rejection?
- Are further privacy, re-identification, or memorization tests necessary?
- What, if any, distribution license is appropriate?

No license has been selected, no rights conclusion has been reached, and no release
has been approved. This document is an engineering review aid and not legal advice.
