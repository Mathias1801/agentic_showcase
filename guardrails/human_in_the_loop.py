"""
human_in_the_loop.py

The "human-in-the-loop checkpoint before finalizing any significant
finding" guardrail. Deliberately rule-based, not LLM-based: asking a model
to self-report "is this a significant/causal claim I'm making?" has the
same self-judging problem the project is designed to avoid, so this is a
plain keyword/pattern scan instead.

A "significant finding" here means the narrative asserts a CAUSE, not just
a pattern. "Margin dropped from 65 to 58" is a pattern. "Margin dropped
because supplier costs rose" is a causal claim, and causal claims are
exactly the kind of thing that should get a human's eyes on it before
reaching a stakeholder — they're the highest-value AND highest-risk
output of the whole pipeline.
"""

import re
from dataclasses import dataclass, field

CAUSAL_PATTERNS = [
    r"\bbecause\b", r"\bdue to\b", r"\bcaused by\b", r"\bcausing\b",
    r"\bdriven by\b", r"\bdrove\b", r"\bresulted? (?:in|from)\b",
    r"\bled to\b", r"\bthe reason\b", r"\bas a result of\b",
    r"\battribut(?:ed|able) to\b", r"\bstems? from\b", r"\bthe cause\b",
    r"\bexplains?\b", r"\baccounts? for\b", r"\bwhich means\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CAUSAL_PATTERNS]


@dataclass
class SignificanceCheck:
    requires_human_review: bool
    matched_phrases: list = field(default_factory=list)


def check_significance(narrative: str) -> SignificanceCheck:
    if not narrative:
        return SignificanceCheck(requires_human_review=False)

    matches = []
    for pattern in _COMPILED:
        for m in pattern.finditer(narrative):
            matches.append(m.group())

    return SignificanceCheck(
        requires_human_review=len(matches) > 0,
        matched_phrases=matches,
    )
