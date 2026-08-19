import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.analysis_agent import AnalysisAgent, compute_grounded_facts
from core.llm_client import LLMResponse


class FakeNarrativeClient:
    """Scripted sequence of narrative responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_narrative(self, question, grounded_facts, prior_error=None):
        self.calls.append({"question": question, "facts": list(grounded_facts),
                            "prior_error": prior_error})
        text = self._responses.pop(0)
        return LLMResponse(text=text, input_tokens=40, output_tokens=15)


def fake_retrieval_result(rows, success=True):
    return SimpleNamespace(success=success, rows=rows)


# ---------------------------------------------------------------------------
# compute_grounded_facts — pure computation, no LLM involved
# ---------------------------------------------------------------------------

def test_single_row_reports_value_directly():
    rows = [{"total": 9089921.4}]
    facts, numbers, analysis_type = compute_grounded_facts(rows)
    assert any("9089921.4" in f for f in facts)
    assert 9089921.4 in numbers
    assert analysis_type == "single_value"


def test_time_series_detects_drop():
    # mirrors the real Home & Garden monthly margin shape (C01)
    rows = [
        {"month": 1, "value": 64.76}, {"month": 2, "value": 64.99},
        {"month": 3, "value": 65.04}, {"month": 4, "value": 58.05},
        {"month": 5, "value": 61.90}, {"month": 6, "value": 64.18},
    ]
    facts, numbers, analysis_type = compute_grounded_facts(rows)
    combined = " ".join(facts)
    assert "largest single-step drop" in combined
    # the march->april drop (65.04 -> 58.05) should be captured
    assert 65.04 in numbers and 58.05 in numbers
    assert analysis_type == "time_series"


def test_categorical_ranking_detects_top_and_bottom():
    # mirrors margin_by_category shape (B06)
    rows = [
        {"category": "Electronics", "value": 61.28},
        {"category": "Home & Garden", "value": 63.66},
        {"category": "Office Supplies", "value": 65.01},
        {"category": "Apparel", "value": 69.41},
        {"category": "Sporting Goods", "value": 70.11},
    ]
    facts, numbers, analysis_type = compute_grounded_facts(rows)
    combined = " ".join(facts)
    assert "Electronics" in combined and "Sporting Goods" in combined
    assert 61.28 in numbers and 70.11 in numbers
    assert analysis_type == "ranking"


def test_empty_rows_returns_no_facts():
    facts, numbers, analysis_type = compute_grounded_facts([])
    assert facts == []
    assert numbers == set()
    assert analysis_type == "no_data"


# ---------------------------------------------------------------------------
# AnalysisAgent — grounding retry loop
# ---------------------------------------------------------------------------

def test_grounded_narrative_accepted_first_try():
    rows = [{"month": 3, "value": 65.04}, {"month": 4, "value": 58.05}]
    fake = FakeNarrativeClient([
        "Margin dropped from 65.04 in month 3 to 58.05 in month 4."
    ])
    agent = AnalysisAgent(llm_client=fake)
    result = agent.analyze("Why did margin drop?", fake_retrieval_result(rows))

    assert result.success
    assert result.grounded
    assert result.attempts == 1


def test_ungrounded_narrative_triggers_retry_then_succeeds():
    rows = [{"month": 3, "value": 65.04}, {"month": 4, "value": 58.05}]
    fake = FakeNarrativeClient([
        "Margin dropped 40% due to a supplier issue.",  # invented number
        "Margin dropped from 65.04 to 58.05.",  # corrected, grounded
    ])
    agent = AnalysisAgent(llm_client=fake)
    result = agent.analyze("Why did margin drop?", fake_retrieval_result(rows))

    assert result.success
    assert result.attempts == 2
    assert fake.calls[1]["prior_error"] is not None


def test_persistently_ungrounded_narrative_is_rejected():
    rows = [{"month": 3, "value": 65.04}, {"month": 4, "value": 58.05}]
    fake = FakeNarrativeClient([
        "Margin dropped 40% due to a supplier issue.",
        "Margin cratered by 75% because of a currency shock.",
    ])
    agent = AnalysisAgent(llm_client=fake)
    result = agent.analyze("Why did margin drop?", fake_retrieval_result(rows))

    assert not result.success
    assert not result.grounded
    assert result.narrative is None
    # the facts themselves should still be available even though no
    # narrative was accepted
    assert len(result.facts) > 0


def test_failed_retrieval_short_circuits():
    agent = AnalysisAgent(llm_client=FakeNarrativeClient([]))
    result = agent.analyze("Anything", fake_retrieval_result([], success=False))

    assert not result.success
    assert "Retrieval did not succeed" in result.error
