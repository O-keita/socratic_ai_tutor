"""
Unit tests for the per-session question cap in main.py

Covers:
  - _tutor_turns_used counts assistant turns only
  - _check_session_limit allows requests under the cap
  - _check_session_limit raises HTTP 403 with a friendly message at the cap
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # main.py's lifespan only warns (not raises) without these in dev, but
    # importing main shouldn't require network access or a real Groq key.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    yield


def _import_main():
    import main
    return main


class TestTutorTurnsUsed:
    def test_no_history_is_zero(self):
        main = _import_main()
        assert main._tutor_turns_used(None) == 0
        assert main._tutor_turns_used([]) == 0

    def test_counts_only_assistant_messages(self):
        main = _import_main()
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what is overfitting?"},
            {"role": "assistant", "content": "what have you tried?"},
        ]
        assert main._tutor_turns_used(history) == 2

    def test_ignores_unknown_roles(self):
        main = _import_main()
        history = [
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "hi"},
        ]
        assert main._tutor_turns_used(history) == 1


class TestCheckSessionLimit:
    def test_allows_under_cap(self):
        main = _import_main()
        history = [{"role": "assistant", "content": "x"}] * (main._MAX_SESSION_QUESTIONS - 1)
        main._check_session_limit(history)  # should not raise

    def test_blocks_at_cap(self):
        main = _import_main()
        history = [{"role": "assistant", "content": "x"}] * main._MAX_SESSION_QUESTIONS
        with pytest.raises(HTTPException) as exc_info:
            main._check_session_limit(history)
        assert exc_info.value.status_code == 403
        assert str(main._MAX_SESSION_QUESTIONS) in exc_info.value.detail

    def test_blocks_over_cap(self):
        main = _import_main()
        history = [{"role": "assistant", "content": "x"}] * (main._MAX_SESSION_QUESTIONS + 3)
        with pytest.raises(HTTPException) as exc_info:
            main._check_session_limit(history)
        assert exc_info.value.status_code == 403
