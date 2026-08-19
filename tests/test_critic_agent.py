import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.critic_agent import CriticAgent
from core.llm_client import LLMResponse


class FakeCriticClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def evaluate(self, question, grounded_facts, narrative):
        self.calls.append({"question": question, "facts": list(grounded_facts),
                            "narrative": narrative})
        return LLMResponse(text=self.response_text, input_tokens=100, output_tokens=50)


def fake_analysis_result(facts, narrative, success=True, analysis_type="time_series"):
    return SimpleNamespace(success=success, facts=facts, narrative=narrative,
                            analysis_type=analysis_type)


def fake_retrieval_result(row_count):
    return SimpleNamespace(row_count=row_count)


GOOD_VERDICT = json.dumps({
    "answers_question": True,
    "unsupported_claims": [],
    "confidence": "high",
    "notes": "Narrative is fully backed by the facts.",
})

BAD_VERDICT = json.dumps({
    "answers_question": True,
    "unsupported_claims": ["a 40% cost spike"],
    "confidence": "low",
    "notes": "The 40% figure isn't in the facts.",
})


def test_well_grounded_high_confidence_narrative_passes():
    facts = ["From month=3 to month=4, value went from 65.04 to 58.05."]
    narrative = "Value dropped from 65.04 in month 3 to 58.05 in month 4."
    agent = CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT))
    result = agent.review(
        "Why did the value drop?",
        fake_analysis_result(facts, narrative),
        fake_retrieval_result(row_count=12),
    )

    assert result.passed
    assert result.grounding_ok
    assert result.confidence_ok
    assert not result.requires_human_review  # no causal language used


def test_causal_narrative_flagged_for_human_review_even_if_passed():
    facts = ["From month=3 to month=4, value went from 65.04 to 58.05."]
    narrative = "Value dropped from 65.04 to 58.05 because of a cost increase."
    agent = CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT))
    result = agent.review(
        "Why did the value drop?",
        fake_analysis_result(facts, narrative),
        fake_retrieval_result(row_count=12),
    )

    assert result.requires_human_review
    assert any("because" in p.lower() for p in result.significant_phrases)


def test_low_row_count_fails_confidence_check_for_trend_claims():
    # a single row can't support a trend/comparison claim -- this SHOULD fail
    facts = ["value = 65.04"]
    narrative = "The value is 65.04."
    agent = CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT))
    result = agent.review(
        "What was the value?",
        fake_analysis_result(facts, narrative, analysis_type="time_series"),
        fake_retrieval_result(row_count=1),
    )

    assert not result.passed
    assert not result.confidence_ok
    assert result.needs_clarification


def test_single_value_result_not_penalized_for_low_row_count():
    # regression test: found via a real pipeline run against "what was
    # total gross revenue in 2024?" -- one row IS the complete, fully
    # confident answer to a simple lookup question. An earlier version of
    # this guardrail treated row_count=1 as low-confidence unconditionally,
    # which meant every simple single-number question got wrongly rejected
    # by the Critic even though there was nothing uncertain about it.
    facts = ["total = 9089921.4"]
    narrative = "Total gross revenue was 9089921.4."
    agent = CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT))
    result = agent.review(
        "What was total gross revenue in 2024?",
        fake_analysis_result(facts, narrative, analysis_type="single_value"),
        fake_retrieval_result(row_count=1),
    )

    assert result.confidence_ok
    assert result.confidence_score == 1.0
    assert result.passed


def test_llm_flagged_unsupported_claim_fails_review():
    facts = ["From month=3 to month=4, value went from 65.04 to 58.05."]
    # imagine the Analysis agent's own grounding check somehow missed this
    narrative = "Value dropped from 65.04 to 58.05, a 40% swing driven by market pressure."
    agent = CriticAgent(llm_client=FakeCriticClient(BAD_VERDICT))
    result = agent.review(
        "Why did the value drop?",
        fake_analysis_result(facts, narrative),
        fake_retrieval_result(row_count=12),
    )

    assert not result.passed
    assert result.llm_verdict["unsupported_claims"] == ["a 40% cost spike"]


def test_independent_grounding_catches_what_analysis_missed():
    # Analysis agent claims it's grounded, but the Critic's own independent
    # check should catch a number that isn't actually backed by the facts
    facts = ["From month=3 to month=4, value went from 65.04 to 58.05."]
    narrative = "Value dropped from 65.04 to 58.05, a change of 999 units."
    agent = CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT))
    result = agent.review(
        "Why did the value drop?",
        fake_analysis_result(facts, narrative),
        fake_retrieval_result(row_count=12),
    )

    assert not result.grounding_ok
    assert 999.0 in result.ungrounded_numbers
    assert not result.passed


def test_malformed_llm_json_does_not_crash_and_fails_closed():
    facts = ["value = 65.04"]
    narrative = "The value is 65.04."
    agent = CriticAgent(llm_client=FakeCriticClient("not valid json at all"))
    result = agent.review(
        "What was the value?",
        fake_analysis_result(facts, narrative),
        fake_retrieval_result(row_count=12),
    )

    assert not result.passed
    assert result.llm_verdict is None
    assert result.llm_parse_error is not None


def test_no_narrative_short_circuits():
    agent = CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT))
    result = agent.review(
        "Anything",
        fake_analysis_result([], None, success=False),
        fake_retrieval_result(row_count=12),
    )

    assert not result.passed
    assert result.needs_clarification
