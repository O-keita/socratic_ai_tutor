# socratic_prompts.py
# Templates for Socratic dialogue and questioning

from typing import List, Dict, Optional


class SocraticPromptBuilder:
    """Builds prompts for Socratic-style tutoring responses."""

    SYSTEM_PROMPT = """You are a Socratic AI tutor specializing in data science and machine learning.

RULES:
1. ALWAYS begin your response with a thinking block containing your reasoning.
2. For conceptual questions: respond with ONE guiding question. NEVER give direct answers. If the student is stuck, give a small hint before your question.
3. For code questions: guide the student to write the code themselves through Socratic questioning.
4. For casual messages (greetings, thanks, chitchat): respond warmly and naturally.

Always start with a thinking block. This is mandatory."""



    DIFFICULTY_PROMPTS = {
        "beginner": "Use simple language and provide high scaffolding.",
        "intermediate": "Encourage analytical thinking and deeper reasoning.",
        "advanced": "Challenge complex assumptions and abstract logic."
    }
    
    def build_prompt(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        difficulty: str = "intermediate",
        topic: Optional[str] = None
    ) -> str:
        """Build a ChatML format prompt for Qwen3-0.6B."""
        level = self.DIFFICULTY_PROMPTS.get(difficulty, self.DIFFICULTY_PROMPTS["intermediate"])
        topic_info = f"Topic: {topic}. " if topic else ""
        
        # Build ChatML style prompt
        prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}\n{topic_info}{level}<|im_end|>\n"
        
        # Add limited history
        for msg in conversation_history[-4:]:
            role = "user" if msg["role"] == "user" else "assistant"
            prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
            
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def build_hint_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        context: Optional[str] = None
    ) -> str:
        """Build a prompt for generating a helpful hint."""
        context_note = f" Context: {context}" if context else ""
        
        return f"""Provide ONE brief, encouraging hint to help the student think through the problem.{context_note}

Guidelines:
- Don't give the answer directly
- Point toward a useful approach or perspective
- Be encouraging and supportive
- Keep it to 1-2 sentences

Hint:"""
