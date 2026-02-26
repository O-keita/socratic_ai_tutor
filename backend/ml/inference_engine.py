import re

from .model_loader import model_loader, load_config
from .socratic_prompts import SocraticPromptBuilder
from evaluation.metrics import AdaptiveMetrics


class InferenceEngine:
    def __init__(self):
        self.model = None
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        tutor_cfg = cfg.get("tutor", {})

        # Inference parameters from config (with sensible defaults)
        self.default_max_tokens = model_cfg.get("max_tokens", 256)
        self.temperature = model_cfg.get("temperature", 0.3)
        self.top_p = model_cfg.get("top_p", 0.9)
        self.top_k = model_cfg.get("top_k", 40)
        self.repeat_penalty = model_cfg.get("repeat_penalty", 1.1)
        self.default_difficulty = tutor_cfg.get("default_difficulty", "intermediate")

    def generate_response(
        self,
        user_message: str,
        history: list = None,
        max_tokens: int = None,
    ) -> dict:
        response_raw = self.generate_response_raw(user_message, history, max_tokens)
        content = response_raw["choices"][0]["message"]["content"]

        # Strip <think>...</think> reasoning block (Qwen3 chain-of-thought)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Calculate metrics for the metadata
        # We include the current message in a temporary full history for analysis
        full_history = (history or []) + [{"role": "user", "content": user_message}]
        socratic_index = AdaptiveMetrics.calculate_socratic_index(full_history)
        scaffolding_level = AdaptiveMetrics.recommend_scaffolding_level(full_history)
        sentiment = AdaptiveMetrics.analyze_sentiment(user_message)

        return {
            "response": content,
            "socratic_index": socratic_index,
            "scaffolding_level": scaffolding_level,
            "sentiment": sentiment
        }

    def generate_response_raw(
        self,
        user_message: str,
        history: list = None,
        max_tokens: int = None,
    ) -> dict:
        if self.model is None:
            self.model = model_loader.load_model()

        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # --- Adaptive scaffolding ------------------------------------------
        # Use conversation history to recommend a difficulty level dynamically.
        difficulty = AdaptiveMetrics.recommend_scaffolding_level(history or [])
        difficulty_hint = SocraticPromptBuilder.DIFFICULTY_PROMPTS.get(
            difficulty,
            SocraticPromptBuilder.DIFFICULTY_PROMPTS[self.default_difficulty],
        )

        system_prompt = (
            f"{SocraticPromptBuilder.SYSTEM_PROMPT}\n\n"
            f"Current student level: {difficulty}. {difficulty_hint}"
        )
        # ------------------------------------------------------------------

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            # Limit to last 8 messages (4 exchanges) to stay within context window
            for msg in history[-8:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        messages.append({"role": "user", "content": user_message})

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        return response


inference_engine = InferenceEngine()
