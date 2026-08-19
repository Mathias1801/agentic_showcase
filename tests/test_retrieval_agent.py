import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMResponse
from agents.retrieval_agent import RetrievalAgent, DB_PATH, get_schema_text, _infer_relationships

REQUIRES_DB = pytest_skip = None
import pytest

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="business.db not found — run data/generate_synthetic_data.py first",
)


class FakeLLMClient:
    """Returns a scripted sequence of responses, one per call — lets tests
    simulate the self-correction retry loop without a real API call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_sql(self, question, schema_text, prior_error=None):
        self.calls.append({"question": question, "prior_error": prior_error})
        response_text = self._responses.pop(0)
        return LLMResponse(text=response_text, input_tokens=50, output_tokens=20)


def test_successful_query_first_try():
    fake = FakeLLMClient(["SELECT SUM(gross_revenue) AS total FROM fact_sales"])
    agent = RetrievalAgent(llm_client=fake)
    result = agent.answer_subquestion("What was total revenue?")

    assert result.success
    assert result.attempts == 1
    assert result.row_count == 1
    assert "total" in result.columns
    assert result.rows[0]["total"] > 0


def test_self_correction_after_invalid_sql():
    fake = FakeLLMClient([
        "SELECT * FROM users",  # invalid: unknown table
        "SELECT COUNT(*) AS n FROM fact_returns",  # valid retry
    ])
    agent = RetrievalAgent(llm_client=fake)
    result = agent.answer_subquestion("How many returns were there?")

    assert result.success
    assert result.attempts == 2
    assert fake.calls[1]["prior_error"] is not None  # error was fed back
    assert result.rows[0]["n"] > 0


def test_needs_clarification_after_repeated_failure():
    fake = FakeLLMClient([
        "DROP TABLE fact_sales",
        "DELETE FROM fact_sales",
    ])
    agent = RetrievalAgent(llm_client=fake)
    result = agent.answer_subquestion("Some question")

    assert not result.success
    assert result.needs_clarification


def test_token_budget_hard_cutoff():
    # budget smaller than a single call's token usage -> should stop
    # immediately without ever getting a valid result
    fake = FakeLLMClient(["SELECT * FROM users"])  # would fail validation anyway
    agent = RetrievalAgent(llm_client=fake, token_budget=10)
    result = agent.answer_subquestion("Anything")

    assert not result.success
    assert result.needs_clarification


def test_row_cap_enforced():
    fake = FakeLLMClient(["SELECT * FROM fact_sales"])
    agent = RetrievalAgent(llm_client=fake)
    # this would return 100k+ rows uncapped; row cap should kick in
    from guardrails.sql_guardrails import MAX_ROW_LIMIT
    result = agent.answer_subquestion("Show me everything")

    assert result.success
    assert result.row_count <= MAX_ROW_LIMIT


def test_schema_text_includes_exact_category_values():
    # regression test: a real gpt-5.4-nano call generated
    # category = 'Home and Garden' (guessing at the value) when the real
    # stored string is 'Home & Garden' -- valid SQL, zero matching rows,
    # no error. The schema text handed to the model must include the
    # exact stored strings for low-cardinality text columns so it isn't
    # guessing at values it can't see.
    schema = get_schema_text()
    assert "Home & Garden" in schema
    assert "UK & Ireland" in schema
    # high-cardinality columns (individual dates) should NOT be dumped in full
    assert schema.count("2025-") < 20


def test_schema_text_includes_join_keys():
    # regression test: a real gpt-5.4-nano call wrote `s.category` directly
    # on fact_sales (where it doesn't exist -- category lives on
    # dim_product) instead of joining. The schema text must make every
    # fact_* -> dim_* join path explicit so the model isn't guessing at
    # table relationships.
    schema = get_schema_text()
    assert "fact_sales.product_id -> dim_product.product_id" in schema
    assert "fact_sales.region_id -> dim_region.region_id" in schema
    assert "fact_returns.product_id -> dim_product.product_id" in schema


def test_infer_relationships_ignores_dim_to_dim_and_self():
    import sqlite3
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]
    relationships = _infer_relationships(conn, tables)
    conn.close()

    # no dim_* table should appear as the LEFT side of a relationship
    assert not any(rel.split(".")[0].startswith("dim_") for rel in relationships)
    # every relationship should point at a genuinely different table
    assert all(rel.split(".")[0] != rel.split(" -> ")[1].split(".")[0] for rel in relationships)
