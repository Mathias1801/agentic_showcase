# Eval

## golden_questions.py

The golden set: 30 questions against `data/business.db`, in four categories:

| Category             | Count | Purpose |
|-----------------------|-------|---------|
| `simple_aggregation`   | 10    | Retrieval accuracy — single-table/single-join facts |
| `multi_join`           | 8     | Retrieval accuracy — requires joining multiple fact tables |
| `causal_ambiguous`     | 8     | Grounding + reasoning — "why did X happen", tied to the four anomalies in `docs/ground_truth.md` |
| `control_no_anomaly`   | 4     | Hallucination check — surface pattern with NO real injected cause; a good system says so instead of inventing an explanation |

Run `python eval/golden_questions.py` to (re)compute ground truth straight
from `business.db` and write `eval/golden_answers.json`. Ground truth is
computed, not hand-typed, so it can't drift out of sync with the data — if
you regenerate `business.db` with a different seed, rerun this too.

`golden_answers.json` is generated output — regenerate it, don't hand-edit it.

## How this feeds the harness (next step)

For each question, the eval harness (not yet built) should run the full
agent pipeline and score three things:

1. **Retrieval accuracy** — do the SQL/pandas queries the Retrieval agent
   actually ran return the same rows as `computed_ground_truth` for that
   question?
2. **Grounding accuracy** — does every number in the final report trace
   back to something the Retrieval agent actually returned (no invented
   figures)?
3. **LLM-as-judge score** — a model *different* from the generator scores
   the final explanation for quality, using `answer_notes` as the rubric.
   For `causal_ambiguous` questions this should check the answer names the
   actual mechanism (see `docs/ground_truth.md`), not just restates the
   symptom. For `control_no_anomaly` questions the judge should penalize
   any confident invented cause.

## Regenerating

```bash
python eval/golden_questions.py
```
