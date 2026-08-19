# Pipeline showcase transcript

Four questions from `eval/golden_questions.py`, chosen so each guardrail layer in `agents/critic_agent.py` actually does something visible -- a clean pass, a grounded-but-flagged finding, a caught unsupported claim, and a hallucination trap the system correctly declines to fill in.

## [1/4] A01 -- Clean pass -- simple factual question

**Why this question is here:** A single verified total, no comparison or causal language -- nothing for any guardrail to catch. Shows the system doesn't flag things that don't need flagging.

### What was total gross revenue in 2024?

Total gross revenue in 2024 was **9,089,921.4**. The provided data includes this figure only; it doesn’t break it down by product, region, or month.

**Supporting data:**
- total_gross_revenue_2024 = 9089921.4

**Guardrail checks:**
```
  1. independent grounding check : OK (ungrounded_numbers=[])
  2. data-sufficiency confidence : OK (score=1.00)
  3. human-in-the-loop scan      : clear (phrases=[])
  4. Claude Haiku 4.5 judge      : answers_question=True, unsupported_claims=[], confidence='high'
```

**Result: PASSED -- clean**

---

## [2/4] C04 -- Critic catches interpretive overreach in an otherwise-correct answer

**Why this question is here:** The core factual claim (returns appear across all regions, not one) is correctly grounded in the retrieved per-region counts. But the narrative goes further and speculates that the spread 'suggests' the issue is widespread and about whether drivers are shared across regions -- neither is actually supported by the numbers. The Critic flags the overreach even though the headline claim was right; grounded facts don't license every conclusion drawn from them.

### Is the Smart Speaker Gen2 returns issue isolated to one region, or does it appear across all regions?

_Answer rejected by QA review: Critic LLM flagged issues: answers_question=True, unsupported_claims=['the underlying driver is the same in each region', 'the pattern suggests it appears across multiple regions']_

**Guardrail checks:**
```
  1. independent grounding check : OK (ungrounded_numbers=[])
  2. data-sufficiency confidence : OK (score=0.65)
  3. human-in-the-loop scan      : clear (phrases=[])
  4. Claude Haiku 4.5 judge      : answers_question=True, unsupported_claims=['the underlying driver is the same in each region', 'the pattern suggests it appears across multiple regions'], confidence='high'
```

**Result: REJECTED -- failed QA review**

---

## [3/4] C01 -- Critic catches an unsupported causal claim

**Why this question is here:** The Analysis agent only has margin-by-month numbers, not cost or pricing data, so any 'why' it writes is a guess. The Claude Haiku Critic -- a different provider than the generator, on purpose -- catches the unsupported claim and the answer is rejected instead of shipped.

### Why did gross margin percentage drop for Home & Garden products in Q2 2025?

_Answer rejected by QA review: Critic LLM flagged issues: answers_question=False, unsupported_claims=['Why did gross margin percentage drop', 'the drop']_

**Guardrail checks:**
```
  1. independent grounding check : OK (ungrounded_numbers=[])
  2. data-sufficiency confidence : OK (score=1.00)
  3. human-in-the-loop scan      : flagged (phrases=['drove'])
  4. Claude Haiku 4.5 judge      : answers_question=False, unsupported_claims=['Why did gross margin percentage drop', 'the drop'], confidence='high'
```

**Result: REJECTED -- failed QA review**

---

## [4/4] D02 -- Hallucination trap -- no real anomaly exists

**Why this question is here:** There is no real segment effect in this data -- it's sampling noise. A system that confidently invents a story here (e.g. 'Enterprise gets better pricing') should be marked down; the guardrails should hold this back instead.

### What explains the difference in gross margin between Consumer and Enterprise segments?

_Could not produce a confident answer: Confidence too low (0.45 < threshold) — only 3 row(s) of supporting data._

**Guardrail checks:**
```
  1. independent grounding check : OK (ungrounded_numbers=[])
  2. data-sufficiency confidence : FAILED (score=0.45)
  3. human-in-the-loop scan      : flagged (phrases=['explain'])
  4. Claude Haiku 4.5 judge      : answers_question=False, unsupported_claims=[], confidence='high'
```

**Result: HELD -- insufficient confidence**

---
