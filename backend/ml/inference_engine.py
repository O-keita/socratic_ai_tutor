import os
import re
import time

from groq import Groq

from .model_loader import model_loader, load_config
from .socratic_prompts import SocraticPromptBuilder
from evaluation.metrics import AdaptiveMetrics


class InferenceEngine:
    def __init__(self):
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        tutor_cfg = cfg.get("tutor", {})

        self.model_name = model_loader.model_name

        # Inference parameters from config (with sensible defaults)
        self.default_max_tokens = model_cfg.get("max_tokens", 256)
        self.temperature = model_cfg.get("temperature", 0.3)
        self.top_p = model_cfg.get("top_p", 0.9)
        self.default_difficulty = tutor_cfg.get("default_difficulty", "intermediate")

        api_key = os.environ.get("GROQ_API_KEY")
        self._client = Groq(api_key=api_key) if api_key else None

    @property
    def model(self):
        """Truthy once a Groq client is configured — mirrors the old local-model flag."""
        return self._client

    def generate_response(
        self,
        user_message: str,
        history: list = None,
        max_tokens: int = None,
    ) -> dict:
        t0 = time.perf_counter()
        response_raw = self.generate_response_raw(user_message, history, max_tokens)
        response_time_ms = round((time.perf_counter() - t0) * 1000)

        content = response_raw.choices[0].message.content

        # Strip <think>...</think> reasoning block, if the model emits one
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        usage = response_raw.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Calculate metrics for the metadata
        full_history = (history or []) + [{"role": "user", "content": user_message}]
        socratic_index = AdaptiveMetrics.calculate_socratic_index(full_history)
        scaffolding_level = AdaptiveMetrics.recommend_scaffolding_level(full_history)
        sentiment = AdaptiveMetrics.analyze_sentiment(user_message)

        stripped = content.rstrip()
        ends_with_question = stripped.endswith("?") if stripped else False

        return {
            "response": content,
            "socratic_index": socratic_index,
            "scaffolding_level": scaffolding_level,
            "sentiment": sentiment,
            "response_time_ms": response_time_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ends_with_question": ends_with_question,
        }

    def generate_response_raw(
        self,
        user_message: str,
        history: list = None,
        max_tokens: int = None,
    ):
        if self._client is None:
            raise RuntimeError("GROQ_API_KEY is not set — cannot reach Groq API.")

        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # --- Adaptive scaffolding ------------------------------------------
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

        return self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )


inference_engine = InferenceEngine()
