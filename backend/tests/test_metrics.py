"""
Unit tests for evaluation/metrics.py

Covers:
  - calculate_socratic_index: empty, user-only, tutor-only, balanced, mixed
  - recommend_scaffolding_level: beginner / intermediate / advanced thresholds
  - analyze_sentiment: low_confidence, high_confidence, neutral, mixed keywords
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from evaluation.metrics import AdaptiveMetrics


# ---------------------------------------------------------------------------
# calculate_socratic_index
# ---------------------------------------------------------------------------

class TestCalculateSocraticIndex:
    def test_empty_history_returns_0_5(self):
        assert AdaptiveMetrics.calculate_socratic_index([]) == 0.5

    def test_only_user_messages_returns_1(self):
        history = [
            {"role": "user", "content": "what is gradient descent"},
            {"role": "user", "content": "can you explain more"},
        ]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert result == 1.0

    def test_only_tutor_messages_returns_0(self):
        history = [
            {"role": "assistant", "content": "what do you think about it"},
            {"role": "assistant", "content": "have you tried running the code"},
        ]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert result == 0.0

    def test_balanced_conversation(self):
        history = [
            {"role": "user",      "content": "a b c d"},      # 4 words
            {"role": "assistant", "content": "e f g h"},      # 4 words
        ]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert result == pytest.approx(0.5)

    def test_student_contributes_more(self):
        history = [
            {"role": "user",      "content": "word " * 7},    # 7 words
            {"role": "assistant", "content": "word " * 3},    # 3 words
        ]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert result == pytest.approx(0.7)

    def test_tutor_contributes_more(self):
        history = [
            {"role": "user",      "content": "ok"},            # 1 word
            {"role": "assistant", "content": "word " * 9},    # 9 words
        ]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert result == pytest.approx(0.1)

    def test_missing_content_key_treated_as_empty(self):
        history = [{"role": "user"}, {"role": "assistant"}]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert result == 0.5

    def test_multi_turn_conversation(self):
        history = [
            {"role": "user",      "content": "what is overfitting"},
            {"role": "assistant", "content": "what do you think causes it"},
            {"role": "user",      "content": "maybe too many parameters"},
            {"role": "assistant", "content": "good can you think of a fix"},
        ]
        result = AdaptiveMetrics.calculate_socratic_index(history)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# recommend_scaffolding_level
# ---------------------------------------------------------------------------

class TestRecommendScaffoldingLevel:
    def _history_with_ratio(self, student_words: int, tutor_words: int):
        return [
            {"role": "user",      "content": "x " * student_words},
            {"role": "assistant", "content": "x " * tutor_words},
        ]

    def test_returns_beginner_when_student_index_below_0_3(self):
        # student 2 / total 10 = 0.2  → beginner
        history = self._history_with_ratio(2, 8)
        assert AdaptiveMetrics.recommend_scaffolding_level(history) == "beginner"

    def test_returns_advanced_when_student_index_above_0_7(self):
        # student 8 / total 10 = 0.8  → advanced
        history = self._history_with_ratio(8, 2)
        assert AdaptiveMetrics.recommend_scaffolding_level(history) == "advanced"

    def test_returns_intermediate_for_balanced(self):
        # student 5 / total 10 = 0.5  → intermediate
        history = self._history_with_ratio(5, 5)
        assert AdaptiveMetrics.recommend_scaffolding_level(history) == "intermediate"

    def test_returns_intermediate_at_lower_boundary(self):
        # exactly 0.3 → intermediate (not beginner)
        history = self._history_with_ratio(3, 7)
        assert AdaptiveMetrics.recommend_scaffolding_level(history) == "intermediate"

    def test_returns_intermediate_at_upper_boundary(self):
        # exactly 0.7 → intermediate (not advanced)
        history = self._history_with_ratio(7, 3)
        assert AdaptiveMetrics.recommend_scaffolding_level(history) == "intermediate"

    def test_empty_history_returns_intermediate(self):
        # index = 0.5 for empty → intermediate
        assert AdaptiveMetrics.recommend_scaffolding_level([]) == "intermediate"


# ---------------------------------------------------------------------------
# analyze_sentiment
# ---------------------------------------------------------------------------

class TestAnalyzeSentiment:
    def test_confused_keyword_returns_low_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("I am confused about this") == "low_confidence"

    def test_stuck_keyword_returns_low_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("I'm stuck on this problem") == "low_confidence"

    def test_cant_keyword_returns_low_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("I can't figure this out") == "low_confidence"

    def test_hard_keyword_returns_low_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("This is really hard for me") == "low_confidence"

    def test_understand_keyword_returns_high_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("I understand now, thanks") == "high_confidence"

    def test_got_it_keyword_returns_high_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("got it, that makes sense") == "high_confidence"

    def test_makes_sense_returns_high_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("makes sense to me now") == "high_confidence"

    def test_easy_keyword_returns_high_confidence(self):
        assert AdaptiveMetrics.analyze_sentiment("that was easy") == "high_confidence"

    def test_neutral_text_returns_neutral(self):
        assert AdaptiveMetrics.analyze_sentiment("what is gradient descent") == "neutral"

    def test_empty_string_returns_neutral(self):
        assert AdaptiveMetrics.analyze_sentiment("") == "neutral"

    def test_case_insensitive_detection(self):
        assert AdaptiveMetrics.analyze_sentiment("I AM CONFUSED") == "low_confidence"
        assert AdaptiveMetrics.analyze_sentiment("I UNDERSTAND") == "high_confidence"

    def test_confusion_keyword_takes_priority_over_confidence(self):
        # Both keywords present — confusion keywords checked first in code
        text = "I don't know but it makes sense somehow"
        assert AdaptiveMetrics.analyze_sentiment(text) == "low_confidence"
