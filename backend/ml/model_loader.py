import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config helper — shared by model_loader and inference_engine
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "data" / "config.json"


def load_config() -> dict:
    """Load backend/data/config.json. Returns an empty dict if not found."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[model_loader] Could not read config.json: {e}. Using defaults.")
        return {}


class ModelLoader:
    """Holds identity/config for the Groq-hosted chat model.

    No local weights are downloaded or loaded — inference happens via the
    Groq API. This class exists so main.py's admin stats can report which
    model is configured without reaching into inference_engine internals.
    """

    def __init__(self):
        cfg = load_config().get("model", {})

        # Model identity — read from config.json (single source of truth).
        self.model_name = os.environ.get("GROQ_MODEL", cfg.get("model_name", "openai/gpt-oss-120b"))

        # Allow env var override, else config.json, else default
        self.n_ctx = int(os.environ.get("N_CTX", cfg.get("n_ctx", 4096)))


model_loader = ModelLoader()
