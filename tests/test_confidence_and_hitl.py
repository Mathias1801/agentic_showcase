import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.confidence import compute_data_confidence, CONFIDENCE_THRESHOLD
from guardrails.human_in_the_loop import check_significance


# --- confidence ---------------------------------------------------------

def test_single_row_is_low_confidence():
    r = compute_data_confidence(1)
    assert r.score < CONFIDENCE_THRESHOLD
    assert not r.passes_threshold


def test_full_year_of_months_is_high_confidence():
    r = compute_data_confidence(12)
    assert r.score >= CONFIDENCE_THRESHOLD
    assert r.passes_threshold


def test_zero_rows_is_zero_confidence():
    r = compute_data_confidence(0)
    assert r.score == 0.0
    assert not r.passes_threshold


def test_confidence_is_monotonic_in_row_count():
    scores = [compute_data_confidence(n).score for n in [1, 3, 6, 12, 50]]
    assert scores == sorted(scores)


# --- human-in-the-loop ---------------------------------------------------

def test_causal_language_flags_review():
    r = check_significance("Margin dropped because supplier costs increased.")
    assert r.requires_human_review
    assert any("because" in p.lower() for p in r.matched_phrases)


def test_pattern_only_language_does_not_flag_review():
    r = check_significance("Margin dropped from 65.04 in March to 58.05 in April.")
    assert not r.requires_human_review
    assert r.matched_phrases == []


def test_multiple_causal_phrases_all_captured():
    text = "This was driven by higher costs, and led to lower margins as a result of the price lag."
    r = check_significance(text)
    assert r.requires_human_review
    assert len(r.matched_phrases) >= 2


def test_empty_narrative_does_not_flag_review():
    r = check_significance("")
    assert not r.requires_human_review


def test_non_numeric_fabricated_explanation_is_flagged():
    # regression test: this exact phrasing (from the D02 trap eval question)
    # contains no numbers at all, so the numeric grounding check can't catch
    # it -- the rule-based causal scan is what has to flag it instead.
    r = check_significance("Enterprise customers get better negotiated pricing, "
                            "which explains the gap.")
    assert r.requires_human_review
