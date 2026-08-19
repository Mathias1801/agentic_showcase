"""
confidence.py

The "confidence threshold below which the system asks a clarifying
question instead of guessing" guardrail. Confidence here is a proxy for
data sufficiency: how many data points actually back the analysis. A
narrative built on 1-2 rows is much less trustworthy than one built on a
full year of daily data, regardless of how confident the LLM's prose
sounds — so this is a deterministic, row-count-based check, not an LLM
self-report.
"""

from dataclasses import dataclass

CONFIDENCE_THRESHOLD = 0.5

# Row-count -> confidence score breakpoints. Deliberately coarse: this is a
# guardrail, not a precision instrument, and it needs to be inspectable and
# defensible in an interview, not a black box.
_BREAKPOINTS = [
    (1, 0.25),   # a single data point: essentially no ability to see a trend
    (3, 0.45),   # 2-3 points: a trend is visible but noisy
    (6, 0.65),   # a handful of points: reasonable
    (12, 0.85),  # a full seasonal cycle (e.g. 12 months): solid
]
_MAX_CONFIDENCE = 0.95


@dataclass
class ConfidenceResult:
    score: float
    row_count: int
    passes_threshold: bool


def compute_data_confidence(row_count: int) -> ConfidenceResult:
    if row_count <= 0:
        score = 0.0
    else:
        score = _MAX_CONFIDENCE
        for threshold_rows, threshold_score in _BREAKPOINTS:
            if row_count <= threshold_rows:
                score = threshold_score
                break

    return ConfidenceResult(
        score=score,
        row_count=row_count,
        passes_threshold=score >= CONFIDENCE_THRESHOLD,
    )
