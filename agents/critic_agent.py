"""
critic_agent.py

The Critic/QA agent — the piece flagged as most important given the
Clinical Compass thesis experience. It reviews the Analysis agent's output
against the original question and the underlying data before anything is
allowed to reach the Report agent.

Four independent checks, deliberately layered rather than relying on any
one of them alone:

  1. grounding_ok       — re-runs the grounding check independently
                           (guardrails/grounding.py) rather than trusting
                           AnalysisResult.grounded. Defense in depth: if
                           the Analysis agent's own check had a bug, this
                           still catches it.
  2. confidence_ok       — deterministic, row-count-based data-sufficiency
                           check (guardrails/confidence.py). A claim built
                           on 1-2 data points doesn't pass regardless of
                           how confident the prose sounds.
  3. requires_human_review — rule-based causal-language scan
                           (guardrails/human_in_the_loop.py). Any claim
                           asserting WHY something happened (not just
                           THAT it happened) gets flagged for a human
                           checkpoint before finalizing, independent of
                           whether it otherwise "passes".
  4. llm_verdict          — a DIFFERENT PROVIDER (see core/llm_client.py —
                           AnthropicCriticClient, Claude Haiku 4.5 by
                           default, not just a different-sized OpenAI
                           model) reads the question, facts, and narrative
                           and gives a structured verdict: does it answer
                           the question, are there unsupported claims,
                           what's its own confidence read. Cross-provider
                           rather than cross-size matters here because
                           same-family models share training lineage and
                           correlate on systematic blind spots — see the
                           rationale in core/llm_client.py.

`passed` is only true if 1, 2, and the LLM verdict all clear — but even a
passing result can still have requires_human_review=True, since "well
grounded" and "should have a human read it" are different questions.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from core.llm_client import CriticLLMClient
from guardrails.confidence import compute_data_confidence
from guardrails.grounding import check_grounding, extract_numbers
from guardrails.human_in_the_loop import check_significance


@dataclass
class CriticResult:
    question: str
    passed: bool = False
    grounding_ok: bool = False
    ungrounded_numbers: list = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_ok: bool = False
    requires_human_review: bool = False
    significant_phrases: list = field(default_factory=list)
    llm_verdict: Optional[dict] = None
    llm_parse_error: Optional[str] = None
    needs_clarification: bool = False
    error: Optional[str] = None
    tokens_used: int = 0


_INVALID_APOSTROPHE_ESCAPE = re.compile(r"\\'")


def _repair_json(raw_text: str) -> str:
    """Claude Haiku occasionally escapes an apostrophe as \\' inside a JSON
    string (valid in Python/JS string literals, not in JSON -- JSON has no
    \\' escape). That single-character slip otherwise throws away an
    entirely correct, well-formed verdict. \\' never appears in valid JSON,
    so unescaping it here can only repair this known quirk, never mask a
    genuinely different parse failure (truncation, missing keys, etc. still
    raise as before)."""
    return _INVALID_APOSTROPHE_ESCAPE.sub("'", raw_text)


def _parse_llm_verdict(raw_text: str) -> tuple:
    """Returns (verdict_dict_or_None, error_message_or_None)."""
    try:
        verdict = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            verdict = json.loads(_repair_json(raw_text))
        except json.JSONDecodeError as e:
            return None, f"Critic LLM did not return valid JSON: {e}"

    required_keys = {"answers_question", "unsupported_claims", "confidence", "notes"}
    missing = required_keys - set(verdict.keys())
    if missing:
        return None, f"Critic LLM JSON missing keys: {sorted(missing)}"

    return verdict, None


class CriticAgent:
    def __init__(self, llm_client: CriticLLMClient):
        self.llm_client = llm_client

    def review(self, question: str, analysis_result, retrieval_result) -> CriticResult:
        result = CriticResult(question=question)

        if not getattr(analysis_result, "success", False) or not analysis_result.narrative:
            result.error = "No grounded narrative was produced to review."
            result.needs_clarification = True
            return result

        facts = analysis_result.facts
        narrative = analysis_result.narrative

        # --- 1. independent grounding re-check ------------------------------
        # Numbers stated in the question itself (e.g. "...in Q2 2025") are
        # allowed here too, same reasoning as agents/analysis_agent.py --
        # otherwise this independent re-check disagrees with the Analysis
        # agent's own (correct) grounding check for no real reason.
        allowed_numbers = set(extract_numbers(" ".join(facts))) | set(extract_numbers(question))
        grounding = check_grounding(narrative, allowed_numbers)
        result.grounding_ok = grounding.is_grounded
        result.ungrounded_numbers = grounding.ungrounded_numbers

        # --- 2. data-sufficiency confidence ---------------------------------
        # Only meaningful for analyses that actually make a trend/comparison
        # claim (time_series, ranking). A "single_value" result (e.g. "total
        # revenue was X") is a complete, fully confident answer with exactly
        # one row — that's not insufficient data, that's just what the
        # answer looks like. Applying the row-count threshold there would
        # wrongly flag every simple factual question as low-confidence.
        analysis_type = getattr(analysis_result, "analysis_type", "unknown")
        row_count = getattr(retrieval_result, "row_count", 0)
        if analysis_type in ("time_series", "ranking"):
            confidence = compute_data_confidence(row_count)
            result.confidence_score = confidence.score
            result.confidence_ok = confidence.passes_threshold
        else:
            result.confidence_score = 1.0
            result.confidence_ok = True

        # --- 3. human-in-the-loop significance check ------------------------
        significance = check_significance(narrative)
        result.requires_human_review = significance.requires_human_review
        result.significant_phrases = significance.matched_phrases

        # --- 4. LLM judge (different model than the generator) -------------
        llm_response = self.llm_client.evaluate(
            question=question, grounded_facts=facts, narrative=narrative
        )
        result.tokens_used += llm_response.input_tokens + llm_response.output_tokens
        verdict, parse_error = _parse_llm_verdict(llm_response.text)
        result.llm_verdict = verdict
        result.llm_parse_error = parse_error

        llm_ok = (
            verdict is not None
            and verdict.get("answers_question") is True
            and not verdict.get("unsupported_claims")
        )

        if not result.grounding_ok:
            result.error = f"Independent grounding check failed: {result.ungrounded_numbers}"
        elif not result.confidence_ok:
            result.needs_clarification = True
            result.error = (
                f"Confidence too low ({result.confidence_score:.2f} < threshold) — "
                f"only {row_count} row(s) of supporting data."
            )
        elif verdict is None:
            result.error = parse_error
        elif not llm_ok:
            result.error = (
                f"Critic LLM flagged issues: answers_question="
                f"{verdict.get('answers_question')}, "
                f"unsupported_claims={verdict.get('unsupported_claims')}"
            )

        result.passed = (
            result.grounding_ok and result.confidence_ok and llm_ok
        )
        return result
