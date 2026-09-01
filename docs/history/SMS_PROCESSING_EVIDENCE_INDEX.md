# SMS Processing Historical Evidence Index

These artifacts remain accurate evidence for the contracts and runs they measured.
They are not the active product architecture.

| Evidence | Current interpretation |
|---|---|
| `docs/architecture/POCKETFINANCER_ANDROID_BASELINE_PHASE_C.md` | Historical deployed/runtime baseline evidence. |
| `docs/architecture/POCKETFINANCER_ANDROID_PROTOCOL_LAB_PHASE_D.md` | Historical Phase D lab; its no-selection result stands. |
| `docs/experiments/POCKETFINANCER_LFM25_350M_CANDIDATE_PROTOCOL_V1.md` | Controlled comparison supporting grounded selection; protocol superseded for product use. |
| `docs/experiments/EXPERIMENT_CATALOG.md` | Measured model/dataset chronology; rows described as current there are current only within the historical research track. |
| `docs/LFM25_350M_PIPELINE_V2.md` and `docs/LFM25_350M_EXPERIMENT_REPORT.md` | Historical experiment chronology, not the active SMS processing plan. |
| `configs/contracts/pocketfinancer-semantic-v2.schema.json` | Historical Semantic experiment contract; not an active label or runtime output. |
| `lfm25/semantic_v2.py`, Candidate Protocol modules, Phase C/D modules | Compatibility/reproduction code for measured evidence. |
| `docs/history/extraction_program/` and `configs/history/pocketfinancer-extraction-v2*.json` | Superseded plan/status and machine policy preserved outside the active path. |

The path `configs/programs/pocketfinancer-extraction-v2-decision-policy.json` is a
compatibility link to the historical frozen policy. It exists solely so the
hash-locked Phase D evaluator remains byte-reproducible; it is not an active plan.

Historical results must not be rewritten as if they used the current analyzer,
currency context, protected pools, selector contract, or workbench. New evidence
uses descriptive contract IDs and configuration hashes rather than a project
generation label.
