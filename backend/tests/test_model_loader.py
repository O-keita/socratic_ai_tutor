"""
Unit tests for ml/model_loader.py

Covers:
  - load_config: missing file returns {}, valid file parses correctly
  - ModelLoader: model_name/n_ctx from env override, from config, and defaults
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        with patch("ml.model_loader._CONFIG_PATH", tmp_path / "nonexistent.json"):
            from ml.model_loader import load_config
            result = load_config()
        assert result == {}

    def test_valid_config_parsed(self, tmp_path):
        cfg = {"model": {"n_ctx": 2048, "model_name": "llama-3.1-8b-instant"}, "tutor": {}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(cfg))
        with patch("ml.model_loader._CONFIG_PATH", config_file):
            from ml.model_loader import load_config
            result = load_config()
        assert result["model"]["n_ctx"] == 2048
        assert result["model"]["model_name"] == "llama-3.1-8b-instant"

    def test_malformed_json_returns_empty_dict(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{not valid json}")
        with patch("ml.model_loader._CONFIG_PATH", config_file):
            from ml.model_loader import load_config
            result = load_config()
        assert result == {}


# ---------------------------------------------------------------------------
# ModelLoader construction
# ---------------------------------------------------------------------------

class TestModelLoaderInit:
    def _make_loader(self, env=None, config=None):
        from ml.model_loader import ModelLoader
        env = env or {}
        config = config or {}
        with patch.dict(os.environ, env, clear=False), \
             patch("ml.model_loader.load_config", return_value=config):
            return ModelLoader()

    def test_model_name_from_env(self):
        loader = self._make_loader(env={"GROQ_MODEL": "mixtral-8x7b"})
        assert loader.model_name == "mixtral-8x7b"

    def test_model_name_from_config_when_env_unset(self):
        saved = os.environ.pop("GROQ_MODEL", None)
        try:
            loader = self._make_loader(config={"model": {"model_name": "llama-3.1-8b-instant"}})
        finally:
            if saved is not None:
                os.environ["GROQ_MODEL"] = saved
        assert loader.model_name == "llama-3.1-8b-instant"

    def test_model_name_default(self):
        saved = os.environ.pop("GROQ_MODEL", None)
        try:
            loader = self._make_loader(config={})
        finally:
            if saved is not None:
                os.environ["GROQ_MODEL"] = saved
        assert loader.model_name == "openai/gpt-oss-120b"

    def test_n_ctx_from_env(self):
        loader = self._make_loader(env={"N_CTX": "8192"})
        assert loader.n_ctx == 8192

    def test_n_ctx_from_config_when_env_unset(self):
        saved = os.environ.pop("N_CTX", None)
        try:
            loader = self._make_loader(config={"model": {"n_ctx": 2048}})
        finally:
            if saved is not None:
                os.environ["N_CTX"] = saved
        assert loader.n_ctx == 2048
