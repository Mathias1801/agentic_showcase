import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.orchestrator import Orchestrator
from agents.report_agent import ReportAgent
from agents.retrieval_agent import RetrievalAgent
from core.llm_client import LLMResponse


class FakeSQLClient:
    def __init__(self, sql):
        self.sql = sql

    def generate_sql(self, question, schema_text, prior_error=None):
        return LLMResponse(text=self.sql, input_tokens=80, output_tokens=40)


class FakeNarrativeClient:
    def __init__(self, text):
        self.text = text

    def generate_narrative(self, question, grounded_facts, prior_error=None):
        return LLMResponse(text=self.text, input_tokens=100, output_tokens=50)


class FakeCriticClient:
    def __init__(self, verdict):
        self.verdict = verdict

    def evaluate(self, question, grounded_facts, narrative):
        return LLMResponse(text=json.dumps(self.verdict), input_tokens=100, output_tokens=50)


GOOD_VERDICT = {"answers_question": True, "unsupported_claims": [], "confidence": "high",
                "notes": "Backed by the facts."}

MARGIN_SQL = """
    SELECT d.month AS month, ROUND(AVG(s.gross_margin_pct)*100,2) AS value
    FROM fact_sales s JOIN dim_date d ON s.date_id=d.date_id
    JOIN dim_product p ON s.product_id=p.product_id
    WHERE p.category='Home & Garden' AND d.year=2025
    GROUP BY d.month ORDER BY d.month
"""


# --- Report agent ---------------------------------------------------------

def test_report_answered_status_includes_narrative_and_facts():
    analysis = SimpleNamespace(narrative="Value dropped from 65 to 58.", facts=["fact one"])
    critic = SimpleNamespace(passed=True, needs_clarification=False,
                              requires_human_review=False)
    report = ReportAgent().build_report_from_pipeline("Why?", analysis, critic)

    assert report.status == "answered"
    assert report.narrative == "Value dropped from 65 to 58."
    assert report.supporting_facts == ["fact one"]
    assert "Value dropped" in report.to_markdown()


def test_report_flags_human_review_in_markdown():
    analysis = SimpleNamespace(narrative="Dropped because of X.", facts=[])
    critic = SimpleNamespace(passed=True, needs_clarification=False,
                              requires_human_review=True)
    report = ReportAgent().build_report_from_pipeline("Why?", analysis, critic)

    assert report.requires_human_review
    assert "human review" in report.to_markdown().lower()


def test_report_rejected_when_critic_fails():
    critic = SimpleNamespace(passed=False, needs_clarification=False,
                              error="Independent grounding check failed")
    report = ReportAgent().build_report("Why?", critic)

    assert report.status == "rejected"
    assert "rejected" in report.to_markdown().lower()


def test_report_needs_clarification_when_critic_says_so():
    critic = SimpleNamespace(passed=False, needs_clarification=True,
                              error="Confidence too low")
    report = ReportAgent().build_report("Why?", critic)

    assert report.status == "needs_clarification"


# --- Orchestrator (full pipeline, real business.db, fake LLM clients) ----

def test_orchestrator_full_pipeline_answers_grounded_question():
    orchestrator = Orchestrator(
        retrieval_agent=RetrievalAgent(llm_client=FakeSQLClient(MARGIN_SQL)),
        analysis_agent=AnalysisAgent(llm_client=FakeNarrativeClient(
            "Home & Garden margin fell from 65.04 in month 3 to 58.05 in month 4."
        )),
        critic_agent=CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT)),
    )
    report = orchestrator.answer("Why did Home & Garden margin drop in Q2 2025?")

    assert report.status == "answered"
    assert "65.04" in report.narrative
    assert len(report.supporting_facts) > 0


def test_orchestrator_stops_at_retrieval_failure():
    orchestrator = Orchestrator(
        retrieval_agent=RetrievalAgent(llm_client=FakeSQLClient("SELECT * FROM users")),
        analysis_agent=AnalysisAgent(llm_client=FakeNarrativeClient("irrelevant")),
        critic_agent=CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT)),
    )
    report = orchestrator.answer("Some question with an invalid table reference")

    assert report.status in ("needs_clarification", "rejected")


def test_orchestrator_rejects_ungrounded_analysis():
    orchestrator = Orchestrator(
        retrieval_agent=RetrievalAgent(llm_client=FakeSQLClient(MARGIN_SQL)),
        # narrative invents a number that isn't in the retrieved data at all
        analysis_agent=AnalysisAgent(llm_client=FakeNarrativeClient(
            "Margin collapsed by 999 due to a market crash."
        )),
        critic_agent=CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT)),
    )
    report = orchestrator.answer("Why did Home & Garden margin drop in Q2 2025?")

    assert report.status == "rejected"


def test_orchestrator_accepts_single_row_simple_lookup():
    # regression test: "revenue on a single day" is a legitimate
    # single_value question -- one row IS the complete, confident answer.
    # An earlier version of the confidence guardrail treated any
    # single-row result as low-confidence regardless of question shape,
    # which wrongly rejected simple lookups like this one.
    single_row_sql = "SELECT SUM(gross_revenue) AS value FROM fact_sales WHERE date_id = 20250101"
    orchestrator = Orchestrator(
        retrieval_agent=RetrievalAgent(llm_client=FakeSQLClient(single_row_sql)),
        analysis_agent=AnalysisAgent(llm_client=FakeNarrativeClient("Revenue that day was the retrieved figure.")),
        critic_agent=CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT)),
    )
    report = orchestrator.answer("What was revenue on a single day?")

    assert report.status == "answered"


def test_orchestrator_rejects_trend_claim_with_too_few_points():
    # genuine low-confidence case: a trend/comparison question where the
    # retrieval only came back with 2 months -- not enough to trust a
    # trend claim, and this SHOULD be flagged (unlike the single-row case
    # above, which is a different, legitimate shape).
    two_month_sql = """
        SELECT d.month AS month, ROUND(AVG(s.gross_margin_pct)*100,2) AS value
        FROM fact_sales s JOIN dim_date d ON s.date_id=d.date_id
        JOIN dim_product p ON s.product_id=p.product_id
        WHERE p.category='Home & Garden' AND d.year=2025 AND d.month IN (3, 4)
        GROUP BY d.month ORDER BY d.month
    """
    orchestrator = Orchestrator(
        retrieval_agent=RetrievalAgent(llm_client=FakeSQLClient(two_month_sql)),
        analysis_agent=AnalysisAgent(llm_client=FakeNarrativeClient(
            "Margin went from 65.04 in month 3 to 58.05 in month 4."
        )),
        critic_agent=CriticAgent(llm_client=FakeCriticClient(GOOD_VERDICT)),
    )
    report = orchestrator.answer("Why did Home & Garden margin drop in Q2 2025?")

    assert report.status in ("needs_clarification", "rejected")
