"""
report_agent.py

The Report/Narrative agent — the last stop in the pipeline. Purely
formatting, no LLM call: everything it needs (the grounded facts, the
narrative, the Critic's verdict) has already been produced and verified
upstream. Its only job is to package that into a clean, stakeholder-facing
result, and to make sure a rejected or flagged answer never gets
presented as if it were a clean one.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Report:
    question: str
    status: str  # "answered" | "needs_clarification" | "rejected"
    narrative: Optional[str] = None
    supporting_facts: list = field(default_factory=list)
    requires_human_review: bool = False
    review_reason: Optional[str] = None
    notes: Optional[str] = None

    def to_markdown(self) -> str:
        lines = [f"### {self.question}", ""]
        if self.status == "answered":
            lines.append(self.narrative)
            if self.requires_human_review:
                lines.append("")
                lines.append(f"> ⚠️ Flagged for human review: {self.review_reason}")
            lines.append("")
            lines.append("**Supporting data:**")
            for fact in self.supporting_facts:
                lines.append(f"- {fact}")
        elif self.status == "needs_clarification":
            lines.append(f"_Could not produce a confident answer: {self.notes}_")
        else:
            lines.append(f"_Answer rejected by QA review: {self.notes}_")
        return "\n".join(lines)


class ReportAgent:
    def build_report(self, question: str, critic_result) -> Report:
        if getattr(critic_result, "needs_clarification", False):
            return Report(
                question=question,
                status="needs_clarification",
                notes=critic_result.error or "Insufficient data to answer confidently.",
            )

        if not getattr(critic_result, "passed", False):
            return Report(
                question=question,
                status="rejected",
                notes=critic_result.error or "Failed QA review.",
            )

        # find the narrative/facts the Critic reviewed — Critic doesn't
        # store them itself, so pull from what's available on the result
        # via the analysis step that's normally passed alongside it in the
        # orchestrator; callers using build_report directly should prefer
        # build_report_from_pipeline below.
        return Report(
            question=question,
            status="answered",
            requires_human_review=critic_result.requires_human_review,
            review_reason=(
                "narrative makes a causal claim — verify before sharing externally"
                if critic_result.requires_human_review else None
            ),
        )

    def build_report_from_pipeline(self, question: str, analysis_result, critic_result) -> Report:
        """Preferred entry point: takes the Analysis result too, so the
        final report actually has the narrative and facts attached."""
        report = self.build_report(question, critic_result)
        if report.status == "answered":
            report.narrative = analysis_result.narrative
            report.supporting_facts = analysis_result.facts
        return report
