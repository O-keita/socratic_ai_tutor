# model_loader.py
# Loads compressed LLM for offline inference

import json
import os
from pathlib import Path
from typing import Optional

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")


class ModelLoader:
    """Loads and manages the quantized GGUF model for Socratic tutoring."""
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional['Llama'] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.config = self._load_config()
        # Allow environment variable to override config path
        self.model_path = os.getenv('MODEL_PATH', self.config.get('model', {}).get('path', ''))
        self.is_loaded = False
    
    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        config_path = Path(__file__).parent.parent / 'data' / 'config.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config: {e}")
            return {}
    
    def _resolve_model_path(self, path: str) -> str:
        """Resolve model path, handling relative paths."""
        if os.path.isabs(path):
            return path
        # Relative to backend directory
        backend_dir = Path(__file__).parent.parent
        return str(backend_dir / path)
    
    def load_model(self) -> bool:
        """Load the GGUF model into memory."""
        if not LLAMA_AVAILABLE:
            print("llama-cpp-python is not available")
            return False
        
        if self._model is not None:
            print("Model already loaded")
            return True
        
        resolved_path = self._resolve_model_path(self.model_path)
        if not os.path.exists(resolved_path):
            print(f"Model file not found: {resolved_path}")
            return False
        
        try:
            model_config = self.config.get('model', {})
            print(f"Loading model from: {resolved_path}")
            
            # Allow environment variable overrides for hardware optimization
            n_ctx = int(os.getenv('N_CTX', model_config.get('n_ctx', 4096)))
            n_threads = int(os.getenv('N_THREADS', model_config.get('n_threads', 4)))
            n_gpu_layers = int(os.getenv('N_GPU_LAYERS', model_config.get('n_gpu_layers', 0)))

            ModelLoader._model = Llama(
                model_path=resolved_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            self.is_loaded = True
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model(self) -> Optional['Llama']:
        """Get the loaded model instance."""
        if not self.is_loaded:
            self.load_model()
        return ModelLoader._model
    
    def unload_model(self):
        """Unload the model from memory."""
        ModelLoader._model = None
        self.is_loaded = False
        print("Model unloaded")
    
    @property
    def model(self) -> Optional['Llama']:
        """Property to access the model."""
        return self.get_model()


# Global model loader instance
model_loader = ModelLoader()
