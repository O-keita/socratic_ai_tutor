# socratic_prompts.py
# Templates for Socratic dialogue and questioning

from typing import List, Dict, Optional


class SocraticPromptBuilder:
    """Builds prompts for Socratic-style tutoring responses."""

    SYSTEM_PROMPT = """You are a Socratic AI tutor specializing in data science and machine learning.

                Your core teaching philosophy:
                1. NEVER provide direct answers or explanations
                2. ALWAYS respond with thoughtful guiding questions
                3. Help learners discover answers through their own reasoning
                4. Use questions that prompt reflection, analysis, and critical thinking

                Response format:
                - Begin with <think>...</think> to show your pedagogical reasoning
                - Follow with a single, focused question that guides the learner
                - Keep questions concise and targeted to their current understanding level"""



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
