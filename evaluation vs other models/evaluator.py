"""Evaluation: compute semantic similarity between generated and gold answers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EVAL_EMBEDDING_MODEL

# Lazy load evaluation model
_eval_model = None


def get_eval_model():
    global _eval_model
    if _eval_model is None:
        print(f"[Evaluator] Loading evaluation embedding model: {EVAL_EMBEDDING_MODEL}")
        _eval_model = SentenceTransformer(EVAL_EMBEDDING_MODEL)
    return _eval_model


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumed normalized)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_semantic_similarity(generated: str, expected: str) -> float:
    """Encode both strings and return cosine similarity."""
    model = get_eval_model()
    emb_gen = model.encode(generated, normalize_embeddings=True)
    emb_exp = model.encode(expected, normalize_embeddings=True)
    return cosine_similarity(emb_gen, emb_exp)