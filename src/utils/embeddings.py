"""Singleton utility for loading embedding models avoid multiple slow weight materialization cycles."""

from __future__ import annotations

import logging
from typing import Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
    # Silence transformers/sentence_transformers logging by default
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
except ImportError:
    SentenceTransformer = None

# Global cache for loaded models
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _resolve_embedding_device() -> str:
    try:
        from src.aveniq.policy.laya_policy import resolve_device
        dev, _ = resolve_device()
        return dev.type
    except Exception:
        return "cpu"


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """Retrieve or load a singleton instance of an embedding model.

    This prevents expensive redundant loading of model weights (materializing params)
    on every request or agent instantiation.
    """
    if SentenceTransformer is None:
        return None

    if model_name not in _MODEL_CACHE:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            device = _resolve_embedding_device()
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)

    return _MODEL_CACHE[model_name]

