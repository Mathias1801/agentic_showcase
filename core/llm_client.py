"""
llm_client.py

Thin wrappers around LLM APIs. Three clients live here:

- OpenAISQLClient        — NL question -> SQL (used by the Retrieval agent)
- OpenAINarrativeClient  — grounded facts -> natural-language narrative
                           (used by the Analysis agent)
- AnthropicCriticClient  — question + facts + narrative -> structured
                           verdict (used by the Critic/QA agent)

Retrieval and Analysis default to gpt-5.4-nano — as of August 2026 this is
OpenAI's smallest/cheapest model. Override via `model` or the OPENAI_MODEL
env var.

The Critic deliberately uses a DIFFERENT PROVIDER, not just a different
OpenAI model size: Claude Haiku 4.5 (claude-haiku-4-5-20251001), currently
Anthropic's most cost-effective model. This is a direct fix for the
self-judging circularity problem from the Clinical Compass thesis project
(same model as generator and judge).

Why cross-provider rather than just cross-size (e.g. nano generating,
mini judging): models from the same family are typically distilled from a
shared base model, trained on overlapping data, and put through the same
alignment/RLHF process. That means their failure modes correlate — a
smaller sibling model will genuinely catch capability-limited mistakes in
a larger one, but it tends to inherit the same systematic biases and
blind spots, since those come from shared training lineage rather than
model size. Two models from different labs (different pretraining data,
different alignment approach, different institutional priorities around
what "good" looks like) are much less likely to be wrong in the same way
at the same time. For a QA/critic role specifically — where the entire
point is catching what the generator itself couldn't see — that
independence is worth more than using one vendor's whole model lineup.
This is the same principle behind requiring a second, independent human
reviewer rather than asking the original author to review their own work
twice.

An OpenAICriticClient is also included (defaults to gpt-5.4-mini) for
reference/comparison, but AnthropicCriticClient is the recommended default
for the actual pipeline.

All clients are defined against small Protocols so agents can be tested
with a fake client and never need a real API key/network call in unit
tests.
"""

import os
from dataclasses import dataclass
from typing import Optional, Protocol

from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY / ANTHROPIC_API_KEY etc. from a .env
                # file in the project root, if one exists — see .env.example

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
DEFAULT_OPENAI_CRITIC_MODEL = os.environ.get("OPENAI_CRITIC_MODEL", "gpt-5.4-mini")
DEFAULT_ANTHROPIC_CRITIC_MODEL = os.environ.get(
    "ANTHROPIC_CRITIC_MODEL", "claude-haiku-4-5-20251001"
)

SQL_SYSTEM_PROMPT = """You are a SQL generation assistant for a business analytics system.
You write a single read-only SQLite SELECT query that answers the user's question.

Rules:
- Only use tables and columns from the schema provided. Never invent columns.
- Only SELECT statements. Never INSERT, UPDATE, DELETE, DROP, ALTER, or any
  write/DDL/admin statement.
- Exactly one statement. No semicolon-separated multi-statements.
- Prefer explicit column names over SELECT *.
- Return ONLY the SQL query. No explanation, no markdown fences.
"""

NARRATIVE_SYSTEM_PROMPT = """You are a business analyst writing a short, plain-language \
explanation for a stakeholder.

You will be given a list of pre-computed, verified facts (numbers already \
calculated from real data). Rules:
- Use ONLY the numbers given to you. Never calculate, estimate, round \
  differently, or invent any number that is not in the provided facts.
- If the facts don't fully explain the question, say what is and isn't \
  supported by the data rather than filling the gap with a guess.
- Be concise: 2-5 sentences.
- Do not add a root-cause explanation unless one of the provided facts \
  states a cause. If no cause is given, describe the pattern only and say \
  the cause is not established by this data.
"""

CRITIC_SYSTEM_PROMPT = """You are a skeptical QA reviewer checking another analyst's \
explanation before it reaches a business stakeholder. You did NOT write the \
explanation and have no stake in it being right — your only job is to find \
problems.

You will be given: the original question, a list of verified facts (numbers \
already independently computed from real data), and a narrative someone else \
wrote using those facts. Check:
1. Does the narrative answer the actual question asked?
2. Does every claim in the narrative trace back to one of the verified facts?
   Flag anything that sounds like an inference, cause, or explanation that \
   ISN'T directly stated in the facts (e.g. the facts show a number changed, \
   but the narrative claims WHY it changed when no fact states a cause).
3. Does the narrative overstate confidence given how much data backs it \
   (e.g. a strong claim from only 1-2 data points)?

Respond with ONLY a JSON object, no markdown fences, no other text, in
exactly this shape:
{"answers_question": true or false,
 "unsupported_claims": ["list of specific phrases from the narrative not backed by the facts, empty list if none"],
 "confidence": "low" or "medium" or "high",
 "notes": "one short sentence explaining your verdict"}
"""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    """Anything with this method can stand in for the real OpenAI client —
    used to inject a fake client in tests. (Retrieval agent's interface.)"""

    def generate_sql(self, question: str, schema_text: str,
                      prior_error: Optional[str] = None) -> LLMResponse:
        ...


class NarrativeLLMClient(Protocol):
    """Analysis agent's interface — same fake-ability for tests."""

    def generate_narrative(self, question: str, grounded_facts: list,
                            prior_error: Optional[str] = None) -> LLMResponse:
        ...


class CriticLLMClient(Protocol):
    """Critic agent's interface — same fake-ability for tests."""

    def evaluate(self, question: str, grounded_facts: list, narrative: str) -> LLMResponse:
        ...


def _require_openai_client(api_key: Optional[str]):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or pass api_key= explicitly."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is required. Install with: pip install openai"
        ) from e
    return OpenAI(api_key=key)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        for fence in ("json", "sql"):
            if text.lower().startswith(fence):
                text = text[len(fence):]
                break
        text = text.strip()
    return text


class OpenAISQLClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self._client = _require_openai_client(api_key)

    def generate_sql(self, question: str, schema_text: str,
                      prior_error: Optional[str] = None) -> LLMResponse:
        user_content = f"Schema:\n{schema_text}\n\nQuestion: {question}"
        if prior_error:
            user_content += (
                f"\n\nYour previous query was rejected for this reason: "
                f"{prior_error}\nGenerate a corrected query."
            )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SQL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        text = _strip_fences(response.choices[0].message.content)
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


class OpenAINarrativeClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self._client = _require_openai_client(api_key)

    def generate_narrative(self, question: str, grounded_facts: list,
                            prior_error: Optional[str] = None) -> LLMResponse:
        facts_text = "\n".join(f"- {f}" for f in grounded_facts)
        user_content = f"Question: {question}\n\nVerified facts:\n{facts_text}"
        if prior_error:
            user_content += (
                f"\n\nYour previous answer included a number not present in "
                f"the verified facts ({prior_error}). Rewrite using ONLY the "
                f"numbers listed above."
            )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        text = _strip_fences(response.choices[0].message.content)
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


class OpenAICriticClient:
    """Reference/comparison implementation — same-provider critic (mini
    judging nano's output). Kept for completeness, but AnthropicCriticClient
    below is the recommended default; see the module docstring for why."""

    def __init__(self, model: str = DEFAULT_OPENAI_CRITIC_MODEL, api_key: Optional[str] = None):
        self.model = model
        self._client = _require_openai_client(api_key)

    def evaluate(self, question: str, grounded_facts: list, narrative: str) -> LLMResponse:
        facts_text = "\n".join(f"- {f}" for f in grounded_facts)
        user_content = (
            f"Question: {question}\n\nVerified facts:\n{facts_text}\n\n"
            f"Narrative to review:\n{narrative}"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        text = _strip_fences(response.choices[0].message.content)
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


def _extract_text_from_anthropic_content(content_blocks) -> str:
    """Anthropic's Messages API response can contain multiple content
    blocks (e.g. non-text blocks interleaved with the actual answer, or
    the answer split across more than one text block). Reading only
    response.content[0].text risks silently getting an empty or partial
    string if the first block isn't the (or isn't all of the) real
    answer — this concatenates every text-type block instead."""
    parts = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


class AnthropicCriticClient:
    """Recommended default for the Critic agent — see the module docstring
    for why a different provider (not just a different OpenAI model size)
    matters here. Defaults to Claude Haiku 4.5, Anthropic's most
    cost-effective model as of August 2026."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_CRITIC_MODEL, api_key: Optional[str] = None):
        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key= explicitly."
            )
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            ) from e
        self._client = Anthropic(api_key=key)

    def evaluate(self, question: str, grounded_facts: list, narrative: str) -> LLMResponse:
        facts_text = "\n".join(f"- {f}" for f in grounded_facts)
        user_content = (
            f"Question: {question}\n\nVerified facts:\n{facts_text}\n\n"
            f"Narrative to review:\n{narrative}"
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0,
        )
        raw_text = _extract_text_from_anthropic_content(response.content)
        text = _strip_fences(raw_text)

        if not text:
            # Don't silently return an empty LLMResponse -- that surfaces
            # later as a confusing generic JSON-parse error with no clue
            # what actually happened at the API level. Surface the real
            # diagnostic info instead: what block types came back, why the
            # response stopped, and how many tokens were used.
            block_types = [getattr(b, "type", "?") for b in response.content]
            raise RuntimeError(
                f"Anthropic response contained no usable text. "
                f"stop_reason={response.stop_reason!r}, "
                f"content_block_types={block_types}, "
                f"input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens}. "
                f"This usually means max_tokens was hit before any text was "
                f"produced (e.g. consumed by a non-text block) — try raising "
                f"max_tokens, or check if the model/account has extended "
                f"thinking enabled by default."
            )

        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

