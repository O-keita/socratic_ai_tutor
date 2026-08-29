"""
Unit tests for ml/inference_engine.py

The Groq client is always mocked — these tests verify:
  - <think>...</think> blocks are stripped from responses
  - ends_with_question flag is set correctly
  - conversation history is capped at 8 messages
  - all expected keys are present in the returned dict
  - Socratic metrics (index, scaffolding, sentiment) are included
  - system prompt is always the first message sent to the model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_engine():
    """Return an InferenceEngine with a mocked Groq client and no disk/config I/O."""
    with patch("ml.model_loader.load_config", return_value={}), \
         patch("ml.inference_engine.load_config", return_value={}), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
        from ml.inference_engine import InferenceEngine
        engine = InferenceEngine()
        engine._client = MagicMock()
    return engine, engine._client


def _mock_completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    """Build a Groq-SDK-shaped chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


# ---------------------------------------------------------------------------
# Think-block stripping
# ---------------------------------------------------------------------------

class TestThinkBlockStripping:
    def test_think_block_removed_from_response(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "<think>internal reasoning here</think>What have you tried so far?"
        )
        result = engine.generate_response("what is overfitting")
        assert "<think>" not in result["response"]
        assert "internal reasoning" not in result["response"]
        assert "What have you tried so far?" in result["response"]

    def test_multiline_think_block_removed(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "<think>\nline one\nline two\n</think>\nCan you explain what you mean?"
        )
        result = engine.generate_response("help")
        assert "<think>" not in result["response"]
        assert "Can you explain" in result["response"]

    def test_response_without_think_block_unchanged(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "What do you think causes this error?"
        )
        result = engine.generate_response("my code crashes")
        assert result["response"] == "What do you think causes this error?"


# ---------------------------------------------------------------------------
# ends_with_question
# ---------------------------------------------------------------------------

class TestEndsWithQuestion:
    def test_true_when_response_ends_with_question_mark(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "What have you tried so far?"
        )
        result = engine.generate_response("I'm stuck")
        assert result["ends_with_question"] is True

    def test_false_when_response_does_not_end_with_question_mark(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "Think about the training data."
        )
        result = engine.generate_response("what is overfitting")
        assert result["ends_with_question"] is False

    def test_trailing_whitespace_ignored(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "What do you think?   "
        )
        result = engine.generate_response("help")
        assert result["ends_with_question"] is True


# ---------------------------------------------------------------------------
# History capping
# ---------------------------------------------------------------------------

class TestHistoryCapping:
    def test_history_capped_at_8_messages(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")

        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(12)   # 12 messages — should be trimmed to 8
        ]
        engine.generate_response("new question", history=long_history)

        call_messages = client.chat.completions.create.call_args[1]["messages"]
        history_and_user = call_messages[1:]
        # 8 history + 1 current user message = 9 total after system
        assert len(history_and_user) == 9

    def test_short_history_sent_in_full(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")

        short_history = [
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        engine.generate_response("new question", history=short_history)

        call_messages = client.chat.completions.create.call_args[1]["messages"]
        history_and_user = call_messages[1:]
        assert len(history_and_user) == 3  # 2 history + 1 current


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------

class TestResponseStructure:
    def test_all_expected_keys_present(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("What do you think?")
        result = engine.generate_response("hello")
        expected_keys = {
            "response", "socratic_index", "scaffolding_level", "sentiment",
            "response_time_ms", "prompt_tokens", "completion_tokens", "ends_with_question",
        }
        assert expected_keys.issubset(result.keys())

    def test_token_counts_from_usage(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion(
            "ok?", prompt_tokens=42, completion_tokens=7
        )
        result = engine.generate_response("hi")
        assert result["prompt_tokens"] == 42
        assert result["completion_tokens"] == 7

    def test_response_time_ms_is_non_negative_int(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")
        result = engine.generate_response("hi")
        assert isinstance(result["response_time_ms"], int)
        assert result["response_time_ms"] >= 0


# ---------------------------------------------------------------------------
# Metrics integration
# ---------------------------------------------------------------------------

class TestMetricsInResponse:
    def test_socratic_index_in_valid_range(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("What do you think?")
        result = engine.generate_response("I understand gradient descent now")
        assert 0.0 <= result["socratic_index"] <= 1.0

    def test_scaffolding_level_valid_value(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")
        result = engine.generate_response("hi")
        assert result["scaffolding_level"] in {"beginner", "intermediate", "advanced"}

    def test_sentiment_valid_value(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")
        result = engine.generate_response("I'm confused about this")
        assert result["sentiment"] in {"low_confidence", "high_confidence", "neutral"}

    def test_confusion_message_gives_low_confidence_sentiment(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")
        result = engine.generate_response("I am stuck and confused")
        assert result["sentiment"] == "low_confidence"

    def test_system_prompt_always_first_message(self):
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _mock_completion("ok?")
        engine.generate_response("hello", history=[
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "response"},
        ])
        call_messages = client.chat.completions.create.call_args[1]["messages"]
        assert call_messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    def test_raises_when_no_api_key_configured(self):
        with patch("ml.model_loader.load_config", return_value={}), \
             patch("ml.inference_engine.load_config", return_value={}), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            from ml.inference_engine import InferenceEngine
            engine = InferenceEngine()
            import pytest
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                engine.generate_response_raw("hello")
