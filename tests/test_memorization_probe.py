from scripts.probe_lfm25_memorization import (
    _completion_overlap,
    _membership_auc,
)


def test_membership_auc_uses_lower_loss_as_member_signal() -> None:
    assert _membership_auc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert _membership_auc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert _membership_auc([0.5], [0.5]) == 0.5


def test_completion_overlap_reports_only_aggregate_signals() -> None:
    matching = _completion_overlap(
        "violet maple river stone extra",
        "violet maple river stone ending",
    )
    different = _completion_overlap(
        "copper amber",
        "violet maple river stone ending",
    )

    assert matching["verbatim_next_words"] is True
    assert matching["has_shared_rare_ngram"] is True
    assert matching["ngram_jaccard"] > 0
    assert different["verbatim_next_words"] is False
    assert different["has_shared_rare_ngram"] is False
