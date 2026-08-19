"""
demo.py

Showcase run for the pipeline: four questions from eval/golden_questions.py,
deliberately chosen so each guardrail layer actually does something visible
instead of just running one happy-path question.

  A01  simple_aggregation   -> clean pass, nothing to flag
  C04  causal_ambiguous     -> the factual answer is right, but the narrative
                                oversteps into unsupported interpretation ->
                                the Critic catches the overreach even though
                                the core claim was correct
  C01  causal_ambiguous     -> the LLM's "why" isn't backed by any retrieved
                                fact -> the cross-provider Critic (Claude Haiku
                                4.5) rejects it
  D02  control_no_anomaly   -> trap question, no real effect in the data ->
                                guardrails hold it back instead of inventing
                                a confident story

These were chosen after actually running the golden set and watching what
the guardrails did in practice, not written to guarantee a specific outcome
-- see the "why" field on each entry for what's actually being demonstrated.

Unlike ask.py, this drives the four agents directly instead of going through
Orchestrator, so it can print each guardrail's own verdict (grounding,
confidence, human-in-the-loop, LLM judge) alongside the final report -- that
per-check breakdown is the actual point of a demo about quality enforcement,
not just the end result.

Run:
    python demo.py            # print to stdout only
    python demo.py --save     # also write docs/demo_transcript.md
"""

import argparse

from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.report_agent import Report, ReportAgent
from agents.retrieval_agent import RetrievalAgent
from core.llm_client import AnthropicCriticClient, OpenAINarrativeClient, OpenAISQLClient

CURATED = [
    {
        "id": "A01",
        "label": "Clean pass -- simple factual question",
        "question": "What was total gross revenue in 2024?",
        "why": "A single verified total, no comparison or causal language -- "
               "nothing for any guardrail to catch. Shows the system doesn't "
               "flag things that don't need flagging.",
    },
    {
        "id": "C04",
        "label": "Critic catches interpretive overreach in an otherwise-correct answer",
        "question": "Is the Smart Speaker Gen2 returns issue isolated to one "
                     "region, or does it appear across all regions?",
        "why": "The core factual claim (returns appear across all regions, not "
               "one) is correctly grounded in the retrieved per-region counts. "
               "But the narrative goes further and speculates that the spread "
               "'suggests' the issue is widespread and about whether drivers "
               "are shared across regions -- neither is actually supported by "
               "the numbers. The Critic flags the overreach even though the "
               "headline claim was right; grounded facts don't license every "
               "conclusion drawn from them.",
    },
    {
        "id": "C01",
        "label": "Critic catches an unsupported causal claim",
        "question": "Why did gross margin percentage drop for Home & Garden "
                     "products in Q2 2025?",
        "why": "The Analysis agent only has margin-by-month numbers, not "
               "cost or pricing data, so any 'why' it writes is a guess. "
               "The Claude Haiku Critic -- a different provider than the "
               "generator, on purpose -- catches the unsupported claim and "
               "the answer is rejected instead of shipped.",
    },
    {
        "id": "D02",
        "label": "Hallucination trap -- no real anomaly exists",
        "question": "What explains the difference in gross margin between "
                     "Consumer and Enterprise segments?",
        "why": "There is no real segment effect in this data -- it's sampling "
               "noise. A system that confidently invents a story here (e.g. "
               "'Enterprise gets better pricing') should be marked down; the "
               "guardrails should hold this back instead.",
    },
]


def build_agents():
    return (
        RetrievalAgent(llm_client=OpenAISQLClient()),
        AnalysisAgent(llm_client=OpenAINarrativeClient()),
        CriticAgent(llm_client=AnthropicCriticClient()),
        ReportAgent(),
    )


def _status_badge(report, critic_result) -> str:
    if report.status == "answered" and not report.requires_human_review:
        return "PASSED -- clean"
    if report.status == "answered" and report.requires_human_review:
        return "PASSED -- held for human review"
    if report.status == "needs_clarification":
        return "HELD -- insufficient confidence"
    if critic_result is not None and not critic_result.grounding_ok:
        return "REJECTED -- failed independent grounding check"
    return "REJECTED -- failed QA review"


def _diagnostics_lines(critic_result) -> list:
    if critic_result is None:
        return ["  (retrieval or analysis failed before the Critic ran)"]
    verdict = critic_result.llm_verdict
    verdict_line = (
        f"answers_question={verdict.get('answers_question')}, "
        f"unsupported_claims={verdict.get('unsupported_claims')}, "
        f"confidence={verdict.get('confidence')!r}"
        if verdict is not None
        else f"(unparseable: {critic_result.llm_parse_error})"
    )
    return [
        f"  1. independent grounding check : {'OK' if critic_result.grounding_ok else 'FAILED'} "
        f"(ungrounded_numbers={critic_result.ungrounded_numbers})",
        f"  2. data-sufficiency confidence : {'OK' if critic_result.confidence_ok else 'FAILED'} "
        f"(score={critic_result.confidence_score:.2f})",
        f"  3. human-in-the-loop scan      : "
        f"{'flagged' if critic_result.requires_human_review else 'clear'} "
        f"(phrases={critic_result.significant_phrases})",
        f"  4. Claude Haiku 4.5 judge      : {verdict_line}",
    ]


def run_demo(save: bool = False):
    retrieval_agent, analysis_agent, critic_agent, report_agent = build_agents()

    transcript = [
        "# Pipeline showcase transcript",
        "",
        "Four questions from `eval/golden_questions.py`, chosen so each guardrail "
        "layer in `agents/critic_agent.py` actually does something visible -- a "
        "clean pass, a grounded-but-flagged finding, a caught unsupported claim, "
        "and a hallucination trap the system correctly declines to fill in.",
        "",
    ]

    for i, item in enumerate(CURATED, 1):
        header = f"[{i}/{len(CURATED)}] {item['id']} -- {item['label']}"
        print("=" * 100)
        print(header)
        print(f"Q: {item['question']}")
        print("-" * 100)

        retrieval_result = retrieval_agent.answer_subquestion(item["question"])
        critic_result = None

        if not retrieval_result.success:
            report = Report(
                question=item["question"],
                status="needs_clarification" if retrieval_result.needs_clarification else "rejected",
                notes=retrieval_result.error or "Could not retrieve data for this question.",
            )
        else:
            analysis_result = analysis_agent.analyze(item["question"], retrieval_result)
            if not analysis_result.success:
                report = Report(
                    question=item["question"],
                    status="rejected",
                    notes=analysis_result.error or "Could not produce a grounded analysis.",
                )
            else:
                critic_result = critic_agent.review(item["question"], analysis_result, retrieval_result)
                report = report_agent.build_report_from_pipeline(item["question"], analysis_result, critic_result)

        badge = _status_badge(report, critic_result)
        diagnostics = _diagnostics_lines(critic_result)

        print(report.to_markdown())
        print("-" * 100)
        print("Guardrail checks:")
        print("\n".join(diagnostics))
        print(f"\nWhy this question is here: {item['why']}")
        print(f"Result: {badge}")
        print()

        transcript += [
            f"## {header}",
            "",
            f"**Why this question is here:** {item['why']}",
            "",
            report.to_markdown(),
            "",
            "**Guardrail checks:**",
            "```",
            *diagnostics,
            "```",
            "",
            f"**Result: {badge}**",
            "",
            "---",
            "",
        ]

    if save:
        out_path = "docs/demo_transcript.md"
        with open(out_path, "w") as f:
            f.write("\n".join(transcript))
        print(f"Saved transcript to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="also write docs/demo_transcript.md")
    args = parser.parse_args()
    run_demo(save=args.save)
