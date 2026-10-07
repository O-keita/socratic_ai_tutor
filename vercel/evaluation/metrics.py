# metrics.py
# Evaluation metrics for tutoring sessions and adaptive scaffolding logic

from typing import List, Dict, Any

class AdaptiveMetrics:
    """Calculates engagement and understanding metrics to adjust tutor behavior."""
    
    @staticmethod
    def calculate_socratic_index(history: List[Dict[str, str]]) -> float:
        """
        Measures how much the student is contributing vs the tutor.
        Scale: 0.0 to 1.0 (Higher means student is providing more input).
        """
        if not history:
            return 0.5
            
        student_words = 0
        tutor_words = 0
        
        for msg in history:
            word_count = len(msg.get("content", "").split())
            if msg.get("role") == "user":
                student_words += word_count
            else:
                tutor_words += word_count
                
        total = student_words + tutor_words
        return student_words / total if total > 0 else 0.5

    @staticmethod
    def recommend_scaffolding_level(history: List[Dict[str, str]]) -> str:
        """
        Heuristic to recommend a difficulty/scaffolding level.
        Returns: 'beginner', 'intermediate', or 'advanced'
        """
        index = AdaptiveMetrics.calculate_socratic_index(history)
        
        # If student contributions are very low, provide more scaffolding (beginner)
        if index < 0.3:
            return "beginner"
        # If student is leading the conversation, push harder (advanced)
        elif index > 0.7:
            return "advanced"
        else:
            return "intermediate"

    @staticmethod
    def analyze_sentiment(text: str) -> str:
        """Simple keyword-based indicator of student confidence."""
        confusion_keywords = ["don't know", "can't", "hard", "confused", "stuck", "help"]
        confidence_keywords = ["understand", "makes sense", "got it", "easy", "clear"]
        
        text_lower = text.lower()
        if any(w in text_lower for w in confusion_keywords):
            return "low_confidence"
        if any(w in text_lower for w in confidence_keywords):
            return "high_confidence"
        return "neutral"
