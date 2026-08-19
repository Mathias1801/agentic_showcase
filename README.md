# Autonomous Business Analytics Agent

Portfolio project: a multi-agent system that answers ambiguous natural-language
business questions against a synthetic multi-table business dataset, with an
evaluation harness, observability, and guardrails.

## Project structure

```
agentic_flow_project/
├── data/
│   ├── generate_synthetic_data.py   # run this first
│   ├── business.db                  # SQLite output (generated)
│   └── csv/                         # same tables as CSV (generated)
├── docs/
│   ├── data_dictionary.md           # schema reference — safe for agents
│   ├── ground_truth.md              # injected anomalies — NOT for agents,
│   │                                 # used to write/grade eval questions
│   ├── demo_transcript.md           # raw transcript from `python demo.py --save`
│   └── index.html                   # GitHub Pages landing page (Settings -> Pages -> /docs)
├── agents/
│   ├── retrieval_agent.py            # NL question -> validated SQL -> results
│   ├── analysis_agent.py             # computed stats + grounded-only narrative
│   ├── critic_agent.py               # independent re-check + confidence + HITL + LLM judge
│   ├── report_agent.py               # formats the Critic-approved output (no LLM call)
│   └── orchestrator.py               # wires all four into one straight-line pipeline
├── core/
│   └── llm_client.py                 # OpenAI (gen: gpt-5.4-nano) + Anthropic (critic: Claude Haiku 4.5)
├── guardrails/
│   ├── sql_guardrails.py             # read-only, schema-bounded SQL validation
│   ├── grounding.py                  # narrative numbers must trace to computed facts
│   ├── confidence.py                 # row-count-based data-sufficiency threshold
│   └── human_in_the_loop.py          # rule-based causal-language -> flag for human review
├── eval/
│   ├── golden_questions.py          # 30 golden questions + ground-truth computation
│   ├── golden_answers.json          # generated — computed ground truth (don't hand-edit)
│   └── README.md
├── demo.py                           # curated live run showing each guardrail actually firing
├── tests/                            # 75 tests, all passing, no API key needed
├── requirements.txt
└── README.md
```

## Status

- [x] Synthetic dataset generator (`data/generate_synthetic_data.py`)
- [x] Golden eval set (30 questions — `eval/golden_questions.py`)
- [ ] Orchestrator/Planner agent
- [x] Data Retrieval agent (`agents/retrieval_agent.py`) — SQL validation, row cap,
      token budget, one self-correction retry, tests
- [x] Analysis agent (`agents/analysis_agent.py`) — deterministic pandas/numpy stats
      only, LLM narrates but every number is checked against computed facts before
      being accepted; retries once if it invents a number, drops the narrative
      (keeps the facts) if it still can't stay grounded
- [x] Report/Narrative agent (`agents/report_agent.py`) — no LLM call, pure formatting;
      guarantees a rejected/low-confidence result can never be presented as a clean
      answer, since `status` is derived from the Critic's verdict, not asserted
- [x] Orchestrator (`agents/orchestrator.py`) — hand-built, straight-line pipeline:
      Retrieval -> Analysis -> Critic -> Report for a single question. Does NOT yet
      do multi-step question decomposition/looping (see below) — that's the next
      increment once this straight-line version is proven out, which it now is:
      62 tests passing, plus an end-to-end run against real `business.db` confirmed
      the full chain produces a correct, properly-flagged report
- [x] Critic/QA agent (`agents/critic_agent.py`) — independently re-checks grounding
      (doesn't trust Analysis's own self-report), data-sufficiency confidence
      threshold, rule-based human-in-the-loop flagging for causal claims, and a
      structured verdict from a DIFFERENT PROVIDER (Claude Haiku 4.5, via
      `AnthropicCriticClient`) than the one that generated the narrative
      (OpenAI gpt-5.4-nano) — see "Why cross-provider, not just cross-size" below
- [x] Guardrails — SQL validation + row cap + token budget (Retrieval), grounding
      check (Analysis + independently re-checked by Critic), confidence threshold
      + human-in-the-loop causal-language flagging (Critic)

**Known limitation found during testing:** numeric grounding alone doesn't catch
a fabricated explanation that contains no numbers (e.g. "Enterprise gets better
pricing" for a gap that's actually just noise — the D02 trap eval question).
Caught in practice by two other independent layers: the confidence threshold
(too few rows) and the rule-based causal-language scan flagging it for human
review. This is the reason the design is layered rather than relying on any
single check.

## Live output showcase

`demo.py` runs four questions from `eval/golden_questions.py` straight
through the real pipeline (real OpenAI + real Claude Haiku 4.5 calls, no
mocks) and prints each guardrail's own verdict next to the final report —
not just whether the answer passed, but *which* of the four independent
checks in `agents/critic_agent.py` fired and why. These weren't picked to
guarantee a specific outcome; they're what happened on an actual run, after
fixing two real bugs the run itself exposed (see below).

```bash
python demo.py            # print to stdout
python demo.py --save     # also write docs/demo_transcript.md
```

| # | Question | Result | What it shows |
|---|----------|--------|----------------|
| A01 | What was total gross revenue in 2024? | **PASSED — clean** | A correct, fully-grounded answer with nothing to flag — the guardrails stay quiet when there's nothing to catch |
| C04 | Is the Smart Speaker Gen2 returns issue isolated to one region, or all regions? | **REJECTED** | The headline claim was correct, but the narrative speculated beyond it ("the pattern suggests...") — the Critic catches overreach even in an otherwise-right answer |
| C01 | Why did gross margin drop for Home & Garden in Q2 2025? | **REJECTED** | The Analysis agent only has margin numbers, not cost/pricing data — any "why" it writes is a guess, and the cross-provider Critic (Claude Haiku 4.5, not the OpenAI model that wrote the narrative) catches it |
| D02 | What explains the margin difference between Consumer and Enterprise segments? | **HELD** | Trap question — there's no real segment effect in the data, just sampling noise. The confidence guardrail holds it back instead of letting a confident-sounding invented cause through |

Full transcript: [`docs/demo_transcript.md`](docs/demo_transcript.md) · styled version: [GitHub Pages site](docs/index.html) (enable in repo Settings → Pages → Deploy from branch → `/docs`).

Building this surfaced two real bugs before it produced a single clean
pass — both fixed, see items 6-7 in "Bugs found from real LLM calls" below.

## Remaining scope

The core four-agent pipeline (Retrieval -> Analysis -> Critic -> Report) is
built, tested, and verified end-to-end against real data AND real LLM calls
(three real bugs found this way are documented below — none were caught by
the 64+ tests built on hand-written fixtures alone, which is itself worth
remembering).

- Orchestrator currently handles one straight-line question, not the
  decompose-into-sub-questions-and-loop planning behavior from the original
  architecture
- Eval harness that actually runs all 30 golden questions through the
  pipeline and scores retrieval/grounding/judge accuracy (the question bank
  and ground truth already exist in `eval/`, this wires them to the agents)
- MLflow run tracking, GitHub Actions CI
- LangGraph reimplementation of the orchestration loop

## Bugs found from real LLM calls (not caught by fixture-based tests)

1. **Grounding false positive on period labels.** A real gpt-5.4-nano
   narrative referred to "Q2 2025"; the number-extraction guardrail pulled
   the `2` out of `Q2` and flagged it as an invented number. Fixed in
   `guardrails/grounding.py` — now matches whole letter+digit tokens and
   discards the entire token if it starts with a letter, rather than a
   naive lookbehind (which still leaked the back half of multi-digit
   labels like `FY25` through as `25`).
2. **Schema didn't expose exact stored values.** The model knew a
   `category` column existed but not that the exact string is
   `'Home & Garden'` (not `'Home and Garden'`) — valid SQL, zero matching
   rows, no error. Fixed in `agents/retrieval_agent.py::get_schema_text` —
   now includes sample distinct values for any TEXT column with <=30
   distinct values.
3. **Confidence guardrail penalized ALL single-row results, including
   correct ones.** "What was total revenue in 2024?" legitimately returns
   one row — that's the complete answer, not insufficient data. The
   guardrail was treating that identically to a trend question that only
   got one data point when it needed several to support a comparison.
   Fixed by adding `analysis_type` to `AnalysisResult` (`single_value` /
   `time_series` / `ranking` / ...) — the row-count confidence threshold
   in `agents/critic_agent.py` now only applies to `time_series` and
   `ranking` analyses, where a trend/comparison claim is actually being
   made. A `single_value` result gets full confidence regardless of row
   count, since one row is what a correct answer to that shape of
   question looks like.
4. **Critic client only read the first content block from Anthropic's
   response.** `response.content[0].text` isn't guaranteed to be the
   whole (or even any part of the) answer — the Messages API can return
   multiple content blocks. This produced an empty string, which then
   failed JSON parsing with `Expecting value: line 1 column 1 (char 0)`
   — a real answer from Claude Haiku 4.5 was being silently discarded.
   Fixed in `core/llm_client.py` via
   `_extract_text_from_anthropic_content()`, which concatenates every
   text-type block instead of trusting block 0. Also bumped `max_tokens`
   500 -> 1024 for headroom.
5. **Schema text never explained how tables join.** gpt-5.4-nano wrote
   `s.category` directly on `fact_sales` (`no such column: s.category`)
   instead of joining to `dim_product` — likely generalizing from
   `fact_marketing_spend`, which genuinely does have its own standalone
   `category` column, so the model had a real (if wrong) precedent to
   follow. Fixed in `agents/retrieval_agent.py` via
   `_infer_relationships()` — dynamically infers join keys by naming
   convention (a `dim_*` table's first column is its PK; any other table
   with a same-named column references it) and includes them as an
   explicit "Join keys" section in the schema text, rather than leaving
   the model to guess table relationships from column names alone.
6. **Grounding rejected numbers stated in the question itself.** A
   narrative answering "...in 2025" naturally restates the year — but
   both the Analysis agent's grounding check (`agents/analysis_agent.py`)
   and the Critic's independent re-check (`agents/critic_agent.py`) only
   allowed numbers pulled from the computed facts, never from the
   question. Since almost every golden question is scoped to a year or
   quarter, this was blocking most otherwise-correct answers from ever
   reaching a clean pass. Fixed by adding `extract_numbers(question)` to
   the allowed set in both places — restating the question's own
   timeframe isn't an invented figure; a number that appears nowhere in
   the question or the facts still is.
7. **Critic JSON parsing broke on a small-model quoting quirk.** Claude
   Haiku 4.5 occasionally escapes an apostrophe as `\'` inside its verdict
   JSON (valid in Python/JS string literals, not in JSON), which threw
   away an otherwise-correct verdict with a real parse error. Hardened
   `_parse_llm_verdict()` in `agents/critic_agent.py` to repair that one
   specific, unambiguous pattern before falling back to a genuine parse
   failure — it still fails closed (rejects) on any other malformation,
   this only recovers the one known-safe case.

**Also noticed, not yet fixed:** `fact_returns.date_id/product_id/region_id`
are typed `REAL` instead of `INTEGER` like every other table — a classic
`DataFrame.iterrows()` gotcha (a mixed-dtype row gets upcast to a single
dtype, here float64, even though the source column is int64). Doesn't
break query correctness (SQLite compares `5` and `5.0` as equal), but
would need `data/generate_synthetic_data.py` fixed + `business.db`
regenerated to clean up properly.

## Why cross-provider, not just cross-size, for the Critic

The Critic's LLM judge (`AnthropicCriticClient`, Claude Haiku 4.5) is
intentionally a different *provider* from the Retrieval/Analysis generator
(OpenAI gpt-5.4-nano), not just a bigger model in the same family
(gpt-5.4-mini was the first version of this, see git history / earlier
delta zips).

Reasoning: models within one vendor's family are typically distilled from
a shared base model, trained on overlapping data, and put through the
same alignment process. That means their failure modes correlate — a
larger sibling will genuinely catch a smaller model's capability-limited
mistakes, but tends to inherit the same systematic biases, since those
come from shared training lineage rather than raw size. Two models from
different labs (different pretraining data, different RLHF approach,
different institutional judgment calls about what counts as a supported
claim) are much less likely to be wrong in the same way at the same time.
For a QA/critic role — where the entire point is catching what the
generator itself couldn't see — that independence matters more than
staying within one vendor's lineup. This is the direct fix for the
self-judging circularity problem from the Clinical Compass thesis project,
applied more rigorously than a same-vendor model swap would achieve.

Practical cost of this: the project now needs both `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` set, and both `openai` and `anthropic` SDKs installed
(see `requirements.txt`). An `OpenAICriticClient` (gpt-5.4-mini) is still
in `core/llm_client.py` for reference/comparison if you ever want to run
same-provider vs cross-provider critic scoring side by side on the eval
set — that comparison would itself make an interesting section of the
eventual writeup.
- [ ] Langfuse observability
- [ ] MLflow eval tracking
- [ ] GitHub Actions CI
- [ ] LangGraph reimplementation
- [x] Static GitHub Pages demo export (`docs/index.html`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/generate_synthetic_data.py
python -m pytest tests/ -v
```

To run the agents against real LLMs (not just the tests, which use fake
clients): `export OPENAI_API_KEY=...` for Retrieval/Analysis (defaults to
`gpt-5.4-nano`) and `export ANTHROPIC_API_KEY=...` for the Critic (defaults
to `claude-haiku-4-5-20251001`). Worth re-checking both vendors' current
lineups before assuming these defaults stay accurate long-term — verified
current as of August 2026.

## The dataset

A star schema (5 dimension tables + 4 fact tables) covering 2024-01-01 to
2025-12-31: sales transactions, returns, marketing spend, and finance/ops
costs across 5 regions, 3 channels, 3 customer segments, and 18 products
across 5 categories.

Four realistic, non-obvious causal anomalies are baked into 2025 so that
questions like "why did margin drop in Q2" or "what's driving the returns
spike" have real, discoverable answers rather than being unanswerable noise.
See `docs/data_dictionary.md` for the schema and `docs/ground_truth.md` for
the anomalies (keep that file out of anything the agents can read — it's for
writing/grading the eval set only).
