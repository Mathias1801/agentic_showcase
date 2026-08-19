import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.grounding import check_grounding, extract_numbers


def test_extract_numbers_basic():
    assert extract_numbers("Revenue was 1,234.56 dollars") == [1234.56]


def test_extract_numbers_multiple():
    nums = extract_numbers("From 65.04 to 58.05, a drop of 6.99 points")
    assert nums == [65.04, 58.05, 6.99]


def test_extract_numbers_negative():
    assert extract_numbers("Change was -6.99") == [-6.99]


def test_extract_numbers_ignores_quarter_labels():
    # regression test: a real gpt-5.4-nano response naturally referred to
    # "Q2 2025" and the number extractor was pulling "2" out of "Q2" as if
    # it were a claimed data point, causing a false ungrounded-number
    # rejection in production
    nums = extract_numbers("Margin dropped sharply in Q2 2025, particularly in early Q2.")
    assert nums == [2025.0]


def test_extract_numbers_ignores_period_labels_generally():
    # FY25/H1 are a stricter case: a naive fix that only blocks the FIRST
    # digit of a letter-prefixed token still leaks the back half through
    # (e.g. "FY25" -> 25 or even 5) unless the whole token is discarded
    nums = extract_numbers("Revenue was -6.99 in H1 vs the FY25 target.")
    assert nums == [-6.99]


def test_grounded_narrative_passes():
    narrative = "Margin fell from 65.04 to 58.05, a drop of about 6.99 points."
    allowed = {65.04, 58.05, 6.99}
    result = check_grounding(narrative, allowed)
    assert result.is_grounded
    assert result.ungrounded_numbers == []


def test_rounded_narrative_still_passes_within_tolerance():
    # LLM rounds 65.04 -> 65, 58.05 -> 58 -- should still be considered grounded
    narrative = "Margin fell from about 65 to 58."
    allowed = {65.04, 58.05}
    result = check_grounding(narrative, allowed)
    assert result.is_grounded


def test_invented_number_is_caught():
    narrative = "Margin fell from 65.04 to 58.05, driven by a 40% cost spike."
    allowed = {65.04, 58.05}
    result = check_grounding(narrative, allowed)
    assert not result.is_grounded
    assert 40.0 in result.ungrounded_numbers


def test_empty_narrative_is_trivially_grounded():
    result = check_grounding("No numeric claims here.", {1, 2, 3})
    assert result.is_grounded
