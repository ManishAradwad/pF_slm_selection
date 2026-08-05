from collections import Counter

from lfm25.candidate_curriculum import (
    RELATION_PREFIXES,
    audit_curriculum_overlap,
    generate_candidate_curriculum,
    mix_private_and_curriculum,
)
from lfm25.candidates import extract_candidates, selector_target


def test_curriculum_covers_every_semantic_relation_and_is_candidate_grounded():
    rows = generate_candidate_curriculum(
        seed=9,
        rows_per_relation=2,
        rows_per_negative=1,
    )
    prefixes = Counter()
    for row in rows:
        target = selector_target(row["expected"], extract_candidates(row["sms"]))
        if target["transaction"]:
            prefixes[str(target["counterparty"])[:2]] += 1
    assert set(prefixes) == set(RELATION_PREFIXES)
    assert all(count == 2 for count in prefixes.values())
    assert sum(row["expected"] is None for row in rows) == 6
    pn_rows = [
        row
        for row in rows
        if row["expected"] is not None and row["expected"]["counterparty"] is None
    ]
    assert len(pn_rows) == 2
    assert all("withdrawn" in row["sms"] for row in pn_rows)
    assert all(row["sample_weight"] == 0.2 for row in rows)


def test_curriculum_decontamination_is_aggregate_and_detects_template_overlap():
    rows = generate_candidate_curriculum(
        seed=13,
        rows_per_relation=1,
        rows_per_negative=1,
    )
    clean = audit_curriculum_overlap(
        rows,
        {"clean": [{"sms": "Unrelated fixture notification without a transaction."}]},
    )
    assert clean["exact_overlap_count"] == 0
    assert clean["normalized_template_overlap_count"] == 0
    overlapping = audit_curriculum_overlap(
        rows,
        {"copy": [{"sms": rows[0]["sms"]}]},
    )
    assert overlapping["exact_overlap_count"] == 1
    assert overlapping["normalized_template_overlap_count"] == 1


def test_mixing_is_deterministic_and_retains_provenance():
    curriculum = generate_candidate_curriculum(
        seed=11,
        rows_per_relation=1,
        rows_per_negative=1,
    )
    private = [
        {
            "expected": None,
            "label_tier": "consensus_silver",
            "source": {"record_hash": "f" * 64},
        }
    ]
    first, report = mix_private_and_curriculum(private, curriculum, seed=11)
    second, _ = mix_private_and_curriculum(private, curriculum, seed=11)
    assert first == second
    assert report["private_rows"] == 1
    assert report["curriculum_rows"] == len(curriculum)
