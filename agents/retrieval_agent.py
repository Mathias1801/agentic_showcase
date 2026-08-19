"""
retrieval_agent.py

The Data Retrieval agent. Takes a natural-language sub-question (as would
be handed down by the Orchestrator/Planner agent — not yet built), turns
it into SQL via an LLM, validates the SQL against the guardrails in
sql_guardrails.py, executes it read-only against business.db, and returns
a structured result.

Guardrails applied here (see project brief):
- SQL query validation: read-only, schema-bounded (sql_guardrails.py)
- Row cap on results (MAX_ROW_LIMIT)
- Token/cost budget per question, with a hard cutoff
- One self-correction retry if the LLM's SQL fails validation or fails to
  execute, feeding the error back to the model
- If it still can't produce a valid, executable query after the retry, it
  returns needs_clarification=True instead of guessing

Usage (with the real OpenAI client):
    from core.llm_client import OpenAISQLClient
    from agents.retrieval_agent import RetrievalAgent

    agent = RetrievalAgent(llm_client=OpenAISQLClient())
    result = agent.answer_subquestion("What was total revenue in 2024?")

Usage (with a fake client, no API key needed — see tests/):
    agent = RetrievalAgent(llm_client=my_fake_client)
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.llm_client import LLMClient
from guardrails.sql_guardrails import MAX_ROW_LIMIT, enforce_row_limit, validate_sql

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "business.db"

# Hard per-question token budget. If the LLM calls needed to answer one
# sub-question exceed this, the agent stops and reports needs_clarification
# rather than continuing to spend tokens.
DEFAULT_TOKEN_BUDGET = 4000
MAX_ATTEMPTS = 2  # 1 initial generation + 1 self-correction retry


@dataclass
class RetrievalResult:
    question: str
    sql: Optional[str] = None
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    success: bool = False
    needs_clarification: bool = False
    error: Optional[str] = None
    tokens_used: int = 0
    attempts: int = 0


def _infer_relationships(conn, tables) -> list:
    """Infers join keys by naming convention rather than hardcoding them:
    every dim_* table's FIRST column is treated as its primary key (true
    for this schema), and any OTHER table with a column of that exact
    name is treated as referencing it. Dynamic on purpose, same reasoning
    as the rest of this function — a hand-maintained relationship list
    would silently drift out of sync if the schema ever changes.

    This exists because the schema text previously listed columns with no
    indication of how tables relate — a real production run showed a
    small model (gpt-5.4-nano) writing `s.category` directly on
    fact_sales (where it doesn't exist) instead of joining to
    dim_product, likely generalizing from fact_marketing_spend, which
    DOES have its own standalone category column. Making the join paths
    explicit removes the ambiguity instead of hoping the model infers it."""
    table_columns = {}
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
        table_columns[t] = [c[1] for c in cols]

    dim_primary_keys = {
        t: cols[0] for t, cols in table_columns.items()
        if t.startswith("dim_") and cols
    }

    relationships = []
    for table, cols in table_columns.items():
        if table.startswith("dim_"):
            continue
        for col in cols:
            for dim_table, pk_col in dim_primary_keys.items():
                if col == pk_col and dim_table != table:
                    relationships.append(f"{table}.{col} -> {dim_table}.{pk_col}")
    return relationships


def get_schema_text(db_path: Path = DB_PATH, max_distinct_values: int = 30) -> str:
    """Introspects the actual schema from business.db, so this can never
    drift out of sync with the real tables/columns (unlike a hand-written
    schema description).

    Also includes sample distinct values for low-cardinality TEXT columns
    (category, region_name, channel_name, etc). Without this, the model
    only knows a column exists and has to GUESS the exact stored string —
    e.g. it might write category = 'Home and Garden' when the real value
    is 'Home & Garden', which is valid SQL that silently returns zero
    rows rather than erroring. High-cardinality text columns (like dates)
    are left alone; only columns with <= max_distinct_values are listed.

    Also includes an explicit join-key section (see _infer_relationships)
    — without it, a small model has to guess which tables need joining
    for a given column, and can guess wrong in a way that's syntactically
    valid but semantically broken (e.g. referencing a column that exists
    on a DIFFERENT table than the one queried)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()]
        lines = []
        for table in sorted(tables):
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_parts = []
            for c in cols:
                col_name, col_type = c[1], c[2]
                desc = f"{col_name} ({col_type})"
                if col_type.upper() == "TEXT":
                    distinct_count = conn.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                    ).fetchone()[0]
                    if 0 < distinct_count <= max_distinct_values:
                        values = [r[0] for r in conn.execute(
                            f"SELECT DISTINCT {col_name} FROM {table} ORDER BY {col_name}"
                        ).fetchall()]
                        desc += f" -- exact values: {values}"
                col_parts.append(desc)
            lines.append(f"{table}: {', '.join(col_parts)}")

        relationships = _infer_relationships(conn, tables)
        if relationships:
            lines.append("")
            lines.append("Join keys (a fact_* table's column on the left "
                          "must be joined to the dim_* table on the right "
                          "to access that dim table's other columns, e.g. "
                          "category, region_name, product_name):")
            for rel in relationships:
                lines.append(f"  {rel}")

        return "\n".join(lines)
    finally:
        conn.close()


class RetrievalAgent:
    def __init__(self, llm_client: LLMClient, db_path: Path = DB_PATH,
                 token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.llm_client = llm_client
        self.db_path = db_path
        self.token_budget = token_budget
        self._schema_text = get_schema_text(db_path)

    def answer_subquestion(self, question: str) -> RetrievalResult:
        result = RetrievalResult(question=question)
        prior_error: Optional[str] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts = attempt

            if result.tokens_used >= self.token_budget:
                result.needs_clarification = True
                result.error = (
                    f"Token budget ({self.token_budget}) exhausted before a valid "
                    f"query was produced."
                )
                return result

            llm_response = self.llm_client.generate_sql(
                question=question, schema_text=self._schema_text, prior_error=prior_error
            )
            result.tokens_used += llm_response.input_tokens + llm_response.output_tokens
            candidate_sql = llm_response.text

            validation = validate_sql(candidate_sql)
            if not validation.is_valid:
                prior_error = validation.reason
                result.error = validation.reason
                continue

            try:
                columns, rows, truncated = self._execute(candidate_sql)
            except sqlite3.Error as e:
                prior_error = f"SQL execution error: {e}"
                result.error = prior_error
                continue

            result.sql = candidate_sql
            result.columns = columns
            result.rows = rows
            result.row_count = len(rows)
            result.truncated = truncated
            result.success = True
            result.error = None
            return result

        # exhausted attempts without a valid, executable query
        result.needs_clarification = True
        return result

    def _execute(self, sql: str):
        capped_sql = enforce_row_limit(sql, MAX_ROW_LIMIT)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(capped_sql)
            columns = [d[0] for d in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
            truncated = len(rows) == MAX_ROW_LIMIT
            return columns, rows, truncated
        finally:
            conn.close()
