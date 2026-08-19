import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.llm_client import (
    AnthropicCriticClient,
    OpenAICriticClient,
    OpenAISQLClient,
    _extract_text_from_anthropic_content,
)


def test_anthropic_critic_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicCriticClient()


def test_openai_critic_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICriticClient()


def test_openai_sql_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAISQLClient()


def test_anthropic_critic_client_accepts_explicit_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # should not raise the "not set" error when a key is passed explicitly
    # (may still fail later if the anthropic package isn't installed, which
    # is a separate, clearly-labeled error)
    try:
        AnthropicCriticClient(api_key="test-key-not-real")
    except RuntimeError as e:
        assert "anthropic" in str(e).lower() and "not set" not in str(e).lower()


# --- regression tests: multi-content-block extraction ---------------------
# Found in production: reading only response.content[0].text produced an
# empty string (json.loads then failed with "Expecting value: line 1
# column 1 (char 0)"), because the real answer wasn't necessarily in the
# first content block Anthropic's API returned.

def test_extract_text_skips_empty_leading_block():
    blocks = [
        SimpleNamespace(type="text", text=""),
        SimpleNamespace(type="text", text='{"a": 1}'),
    ]
    assert _extract_text_from_anthropic_content(blocks) == '{"a": 1}'


def test_extract_text_single_block_unaffected():
    blocks = [SimpleNamespace(type="text", text='{"a": 1}')]
    assert _extract_text_from_anthropic_content(blocks) == '{"a": 1}'


def test_extract_text_skips_non_text_blocks():
    blocks = [
        SimpleNamespace(type="thinking", text=None),
        SimpleNamespace(type="text", text='{"a": 1}'),
    ]
    assert _extract_text_from_anthropic_content(blocks) == '{"a": 1}'


def test_extract_text_concatenates_multiple_text_blocks():
    blocks = [
        SimpleNamespace(type="text", text='{"a": '),
        SimpleNamespace(type="text", text='1}'),
    ]
    assert _extract_text_from_anthropic_content(blocks) == '{"a": 1}'


def test_extract_text_empty_content_returns_empty_string():
    assert _extract_text_from_anthropic_content([]) == ""


def test_anthropic_evaluate_raises_clear_diagnostic_on_empty_response(monkeypatch):
    # Regression test for a real failure: the Critic got
    # "Expecting value: line 1 column 1 (char 0)" from json.loads twice in
    # a row, even after fixing multi-block extraction -- meaning the
    # response genuinely had no usable text, not just a block-ordering
    # issue. Rather than let that surface later as an opaque JSON error,
    # evaluate() should raise a clear diagnostic immediately, showing
    # stop_reason and block types so the real cause is visible next time.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    client = AnthropicCriticClient()

    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=50, output_tokens=0),
    )
    monkeypatch.setattr(
        client._client.messages, "create", lambda **kwargs: fake_response
    )

    with pytest.raises(RuntimeError, match="no usable text"):
        client.evaluate("Why?", ["fact one"], "some narrative")
