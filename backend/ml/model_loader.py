import os
import json
import requests
from pathlib import Path
from llama_cpp import Llama
from tqdm import tqdm

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
    def __init__(self):
        cfg = load_config().get("model", {})

        # Corrected repo ID — the old one had a typo ("Socatic" → "Socratic")
        self.repo_id = "Omar-keita/DSML-Socratic-qwen3-0.6B"
        self.filename = "Socratic-Qwen3-0.6-Merged-Quality_Data-752M-Q4_K_M (1).gguf"

        # Allow full override via environment variable (useful in Docker)
        env_path = os.environ.get("MODEL_PATH", "")
        self.model_path = (
            Path(env_path)
            if env_path
            else Path(__file__).parent.parent / "models" / self.filename
        )

        # Model parameters: env var > config.json > hardcoded defaults
        self.n_ctx = int(os.environ.get("N_CTX", cfg.get("n_ctx", 4096)))
        self.n_threads = int(os.environ.get("N_THREADS", cfg.get("n_threads", 4)))
        self.n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", cfg.get("n_gpu_layers", 0)))

        self.model = None

    def download_model(self) -> bool:
        if self.model_path.exists():
            return True

        print("[model_loader] Downloading model from Hugging Face...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Primary repo URL
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"
        response = requests.get(url, stream=True)

        # Fallback: try the typo'd repo name in case the file lives there
        if response.status_code != 200:
            fallback_repo = self.repo_id.replace("Socratic", "Socatic")
            fallback_url = f"https://huggingface.co/{fallback_repo}/resolve/main/{self.filename}"
            print(f"[model_loader] Primary URL failed (HTTP {response.status_code}). Trying fallback...")
            response = requests.get(fallback_url, stream=True)

        if response.status_code != 200:
            print(f"[model_loader] Error: Failed to download model (HTTP {response.status_code}).")
            return False

        total_size = int(response.headers.get("content-length", 0))
        with open(self.model_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=self.filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                pbar.update(size)

        print("[model_loader] Download complete.")
        return True

    def load_model(self) -> Llama:
        if self.model is None:
            if not self.download_model():
                raise RuntimeError("Could not download or locate model file.")

            print(
                f"[model_loader] Loading model from {self.model_path} "
                f"(n_ctx={self.n_ctx}, n_threads={self.n_threads}, n_gpu_layers={self.n_gpu_layers})..."
            )
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format="chatml",
                verbose=False,
            )
            print("[model_loader] Model loaded.")
        return self.model


model_loader = ModelLoader()
