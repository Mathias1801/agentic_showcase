"""
analysis_agent.py

The Analysis agent. Takes the Data Retrieval agent's output and produces a
short natural-language explanation — but every number in that explanation
has to come from deterministic computation on the retrieved rows, never
from the LLM's own arithmetic or imagination.

Two-stage design, which is the whole point of this agent:
  1. compute_grounded_facts() — pure pandas/numpy, no LLM. Produces a list
     of plain-English fact strings (e.g. "value dropped from 65.04 in
     month 3 to 58.05 in month 4") plus the exact numbers behind them.
  2. The LLM only writes prose *about* those facts (NARRATIVE_SYSTEM_PROMPT
     in core/llm_client.py forbids it from introducing new numbers).
     guardrails/grounding.py then checks every number the LLM's narrative
     contains against the numbers from stage 1. If it invented anything,
     the agent retries once with the mismatch fed back, and if it still
     fails, returns the facts without a narrative rather than pass along
     an ungrounded explanation.

This is the direct fix for the self-judging/hallucination problem from
the Clinical Compass thesis project: the numbers are never trusted from
model output, they're computed independently and the model's text is
checked against them.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.llm_client import NarrativeLLMClient
from guardrails.grounding import check_grounding, extract_numbers

MAX_ATTEMPTS = 2
DEFAULT_TOKEN_BUDGET = 4000

SEQUENTIAL_DIMENSIONS = {"month", "quarter", "year", "day", "date_id"}
KNOWN_DIMENSION_NAMES = SEQUENTIAL_DIMENSIONS | {
    "category", "region_name", "channel_name", "segment_name",
    "product_name", "department", "reason_code", "day_of_week",
}


@dataclass
class AnalysisResult:
    question: str
    facts: list = field(default_factory=list)
    narrative: Optional[str] = None
    grounded: bool = False
    ungrounded_numbers: list = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    attempts: int = 0
    tokens_used: int = 0
    analysis_type: str = "unknown"


def _round(x, ndigits=2):
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return x


def _split_dimension_and_metric_columns(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dimension_cols = [c for c in df.columns if c in KNOWN_DIMENSION_NAMES]
    metric_cols = [c for c in numeric_cols if c not in dimension_cols]
    # fall back: if nothing matched the known-dimension set, treat non-numeric
    # columns as dimensions
    if not dimension_cols:
        dimension_cols = [c for c in df.columns if c not in numeric_cols]
    return dimension_cols, metric_cols


def compute_grounded_facts(rows: list) -> tuple:
    """Returns (facts: list[str], numbers: set[float], analysis_type: str)
    computed purely from the retrieved rows — no LLM involved. This is the
    ground truth the narrative gets checked against.

    analysis_type matters for the Critic's confidence check downstream:
    a "single_value" result (e.g. "total revenue was X") is a complete,
    fully confident answer on its own — one row is correct, not a sign of
    insufficient data. A "time_series" or "ranking" result making a trend
    or comparison claim genuinely does need enough rows to back that claim
    up. Treating both the same (as an earlier version of this code did)
    meant simple single-number questions were wrongly flagged as
    low-confidence just for having one row."""
    facts = []
    numbers = set()

    if not rows:
        return facts, numbers, "no_data"

    df = pd.DataFrame(rows)
    dimension_cols, metric_cols = _split_dimension_and_metric_columns(df)

    if not metric_cols:
        # nothing numeric to analyze — just report row count
        facts.append(f"The query returned {len(df)} row(s) with no numeric column to analyze.")
        numbers.add(float(len(df)))
        return facts, numbers, "no_metric"

    metric_col = metric_cols[0]
    df = df.dropna(subset=[metric_col])
    if df.empty:
        facts.append("The query returned no non-null values for the metric column.")
        return facts, numbers, "no_valid_values"

    values = df[metric_col].astype(float)

    # --- single row: just report the value(s) directly -------------------
    if len(df) == 1:
        for col in df.columns:
            v = df.iloc[0][col]
            if pd.api.types.is_numeric_dtype(df[col]):
                v = _round(v)
                numbers.add(float(v))
            facts.append(f"{col} = {v}")
        return facts, numbers, "single_value"

    # --- always include basic aggregate stats -----------------------------
    count, mean, vmin, vmax = len(values), _round(values.mean()), _round(values.min()), _round(values.max())
    facts.append(
        f"Across {count} rows, {metric_col} ranges from {vmin} to {vmax}, averaging {mean}."
    )
    numbers.update([float(count), float(mean), float(vmin), float(vmax)])

    dim_col = dimension_cols[0] if dimension_cols else None
    analysis_type = "aggregate_only"  # fallback if neither branch below applies

    if dim_col and dim_col in SEQUENTIAL_DIMENSIONS:
        analysis_type = "time_series"
        # --- time-series style analysis -----------------------------------
        sorted_df = df.sort_values(dim_col)
        seq_values = sorted_df[metric_col].astype(float).tolist()
        seq_dims = sorted_df[dim_col].tolist()

        first_v, last_v = _round(seq_values[0]), _round(seq_values[-1])
        first_d, last_d = seq_dims[0], seq_dims[-1]
        facts.append(
            f"From {dim_col}={first_d} to {dim_col}={last_d}, {metric_col} went from "
            f"{first_v} to {last_v}."
        )
        numbers.update([float(first_v), float(last_v)])

        # point-to-point deltas — find the single largest drop and rise
        deltas = [seq_values[i] - seq_values[i - 1] for i in range(1, len(seq_values))]
        if deltas:
            max_drop_idx = int(np.argmin(deltas))
            max_rise_idx = int(np.argmax(deltas))
            drop = _round(deltas[max_drop_idx])
            rise = _round(deltas[max_rise_idx])
            if drop < 0:
                facts.append(
                    f"The largest single-step drop was {drop} ({metric_col} went from "
                    f"{_round(seq_values[max_drop_idx])} at {dim_col}={seq_dims[max_drop_idx]} to "
                    f"{_round(seq_values[max_drop_idx + 1])} at {dim_col}={seq_dims[max_drop_idx + 1]})."
                )
                numbers.update([float(drop), float(_round(seq_values[max_drop_idx])),
                                 float(_round(seq_values[max_drop_idx + 1]))])
            if rise > 0 and max_rise_idx != max_drop_idx:
                facts.append(
                    f"The largest single-step rise was {rise} ({metric_col} went from "
                    f"{_round(seq_values[max_rise_idx])} at {dim_col}={seq_dims[max_rise_idx]} to "
                    f"{_round(seq_values[max_rise_idx + 1])} at {dim_col}={seq_dims[max_rise_idx + 1]})."
                )
                numbers.update([float(rise), float(_round(seq_values[max_rise_idx])),
                                 float(_round(seq_values[max_rise_idx + 1]))])

        # simple outlier flag: points more than 1.5 std dev from the mean
        std = float(np.std(seq_values))
        if std > 0:
            for d, v in zip(seq_dims, seq_values):
                if abs(v - values.mean()) > 1.5 * std:
                    rv = _round(v)
                    facts.append(
                        f"{dim_col}={d} is a statistical outlier for {metric_col}: "
                        f"{rv} vs an average of {mean} (std dev {_round(std)})."
                    )
                    numbers.update([float(rv), float(_round(std))])

    elif dim_col:
        analysis_type = "ranking"
        # --- categorical ranking style analysis ----------------------------
        ranked = df[[dim_col, metric_col]].sort_values(metric_col, ascending=False)
        top_label, top_val = ranked.iloc[0][dim_col], _round(ranked.iloc[0][metric_col])
        bottom_label, bottom_val = ranked.iloc[-1][dim_col], _round(ranked.iloc[-1][metric_col])
        gap = _round(top_val - bottom_val)
        facts.append(
            f"By {metric_col}, {dim_col}='{top_label}' ranks highest at {top_val} and "
            f"{dim_col}='{bottom_label}' ranks lowest at {bottom_val} (gap of {gap})."
        )
        numbers.update([float(top_val), float(bottom_val), float(gap)])

    return facts, numbers, analysis_type


class AnalysisAgent:
    def __init__(self, llm_client: NarrativeLLMClient, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.llm_client = llm_client
        self.token_budget = token_budget

    def analyze(self, question: str, retrieval_result) -> AnalysisResult:
        result = AnalysisResult(question=question)

        if not getattr(retrieval_result, "success", False):
            result.error = "Retrieval did not succeed; nothing to analyze."
            return result

        facts, numbers, analysis_type = compute_grounded_facts(retrieval_result.rows)
        result.facts = facts
        result.analysis_type = analysis_type
        if not facts:
            result.error = "No usable data returned to analyze."
            return result

        # Anything literally stated in the fact text is fair game for the
        # narrative to reference (this also covers dimension labels like
        # "month=3" that compute_grounded_facts's curated `numbers` set
        # might not separately include). Numbers stated in the question
        # itself (e.g. "...in Q2 2025") are also allowed -- restating the
        # question's own timeframe isn't an invented figure, and without
        # this the narrative gets flagged as ungrounded just for echoing
        # back the year/quarter it was asked about.
        allowed_numbers = (
            set(numbers)
            | set(extract_numbers(" ".join(facts)))
            | set(extract_numbers(question))
        )

        prior_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts = attempt

            if result.tokens_used >= self.token_budget:
                result.error = f"Token budget ({self.token_budget}) exhausted."
                return result

            llm_response = self.llm_client.generate_narrative(
                question=question, grounded_facts=facts, prior_error=prior_error
            )
            result.tokens_used += llm_response.input_tokens + llm_response.output_tokens

            grounding = check_grounding(llm_response.text, allowed_numbers)
            if grounding.is_grounded:
                result.narrative = llm_response.text
                result.grounded = True
                result.success = True
                result.error = None
                return result

            prior_error = f"numbers not found in verified facts: {grounding.ungrounded_numbers}"
            result.ungrounded_numbers = grounding.ungrounded_numbers
            result.error = prior_error

        # exhausted attempts: return the grounded facts without an
        # ungrounded narrative rather than pass along invented numbers
        result.narrative = None
        result.grounded = False
        result.success = False
        return result
