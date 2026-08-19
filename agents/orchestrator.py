"""
orchestrator.py

Ties the four agents together for a single business question. Hand-built,
no framework — per the project plan, this gets built by hand first so it
can be understood and defended in interviews, with an optional LangGraph
reimplementation later to also demonstrate framework fluency.

This is deliberately the simplest possible version: one question in, one
Report out, through Retrieval -> Analysis -> Critic -> Report in a fixed
sequence. It does NOT yet do the "decompose a complex question into
sub-questions and loop" planning behavior described in the original
architecture — that's the next increment once this straight-line version
is proven out end to end.
"""

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.report_agent import Report, ReportAgent
from agents.retrieval_agent import RetrievalAgent


class Orchestrator:
    def __init__(self, retrieval_agent: RetrievalAgent, analysis_agent: AnalysisAgent,
                 critic_agent: CriticAgent, report_agent: ReportAgent = None):
        self.retrieval_agent = retrieval_agent
        self.analysis_agent = analysis_agent
        self.critic_agent = critic_agent
        self.report_agent = report_agent or ReportAgent()

    def answer(self, question: str) -> Report:
        retrieval_result = self.retrieval_agent.answer_subquestion(question)

        if not retrieval_result.success:
            return Report(
                question=question,
                status="needs_clarification" if retrieval_result.needs_clarification else "rejected",
                notes=retrieval_result.error or "Could not retrieve data for this question.",
            )

        analysis_result = self.analysis_agent.analyze(question, retrieval_result)

        if not analysis_result.success:
            return Report(
                question=question,
                status="rejected",
                notes=analysis_result.error or "Could not produce a grounded analysis.",
            )

        critic_result = self.critic_agent.review(question, analysis_result, retrieval_result)

        return self.report_agent.build_report_from_pipeline(question, analysis_result, critic_result)
