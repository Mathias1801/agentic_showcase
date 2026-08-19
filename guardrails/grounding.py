"""
grounding.py

The "no free-generated numbers" guardrail for the Analysis agent. All
numeric analysis in this project must come from deterministic computation
on the Retrieval agent's actual output (agents/analysis_agent.py), never
from the LLM. This module checks that any narrative text written by an
LLM doesn't contain a number that isn't traceable back to that computed,
verified set of facts.

This is exactly the check the project brief calls out as the main defense
against the self-judging / hallucination problems from the Clinical
Compass thesis project: don't just trust the model's prose, verify every
number in it against ground truth that was computed independently of the
model.
"""

import re
from dataclasses import dataclass, field

NUMBER_PATTERN = re.compile(r"[A-Za-z]*-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list:
    """Pulls every number out of a piece of text (handles commas and a
    trailing % sign, ignores the % itself since we compare magnitudes).

    Deliberately does NOT match a period label like 'Q2', 'H1', or 'FY25'
    as if it were a claimed number. This matches the whole letter+digit
    token and discards it entirely if it starts with a letter, rather than
    just excluding the first digit — a naive lookbehind-only approach lets
    the back half of a token like 'FY25' leak through as the number 25 (or
    even 5), since the regex engine just retries matching from the next
    digit after the first one is blocked."""
    numbers = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group()
        if not raw or raw[0].isalpha():
            continue  # letter-prefixed token (Q2, H1, FY25, ...) — not a real number
        raw = raw.replace(",", "")
        if raw in ("", "-", "."):
            continue
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


@dataclass
class GroundingResult:
    is_grounded: bool
    ungrounded_numbers: list = field(default_factory=list)
    checked_numbers: list = field(default_factory=list)


def check_grounding(narrative: str, allowed_numbers, tolerance: float = 0.05,
                     abs_tolerance: float = 0.5) -> GroundingResult:
    """
    Verifies every number mentioned in `narrative` is close to at least one
    number in `allowed_numbers` (the facts computed by the Analysis agent
    directly from retrieved data).

    A number counts as grounded if it's within `abs_tolerance` OR within
    `tolerance` relative error of some allowed number — this tolerates the
    LLM rounding "58.05" to "58" or "58.1" without treating that as an
    invented figure, while still catching genuinely made-up numbers.

    Small integers (0-12) are excluded from the check by default via the
    caller filtering allowed narrative content — counts like "3 months" or
    ordinary sentence numbers are common and low-risk; callers that need
    stricter checking can lower abs_tolerance.
    """
    allowed = list(allowed_numbers)
    narrative_numbers = extract_numbers(narrative)
    ungrounded = []

    for n in narrative_numbers:
        grounded = False
        for a in allowed:
            if abs(n - a) <= abs_tolerance:
                grounded = True
                break
            if a != 0 and abs(n - a) / abs(a) <= tolerance:
                grounded = True
                break
        if not grounded:
            ungrounded.append(n)

    return GroundingResult(
        is_grounded=len(ungrounded) == 0,
        ungrounded_numbers=ungrounded,
        checked_numbers=narrative_numbers,
    )
