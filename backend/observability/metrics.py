"""
Observability Metrics Utilities
================================
Pure functions for computing derived metrics. These are intentionally
stateless — they take inputs and return outputs, making them trivially
testable and composable.

WHY these exist:
- `estimate_tokens`: Ollama doesn't expose token counts in its API.
  The 4-char heuristic is standard for English (GPT tokenizers average
  ~4 chars/token). Good enough for cost tracking and trend analysis.
- `compute_confidence`: ChromaDB returns L2 distances (lower = more
  similar). We need a 0-1 score for thresholding and display.
"""

from config import CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using character-based heuristic.
    Standard approximation: ~4 characters per token for English text.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def compute_confidence(distances: list[float]) -> float:
    """
    Convert ChromaDB L2 distances to a 0-1 confidence score.

    Formula: confidence = 1 / (1 + mean(distances))
    - Distance 0 → confidence 1.0 (perfect match)
    - Distance ∞ → confidence 0.0 (no match)

    This sigmoid-like mapping provides intuitive scores:
    - > 0.5: Good retrieval
    - 0.35–0.5: Marginal retrieval
    - < 0.35: Poor retrieval (triggers circuit breaker)
    """
    if not distances:
        return 0.0
    mean_distance = sum(distances) / len(distances)
    return 1.0 / (1.0 + mean_distance)


def distances_to_similarities(distances: list[float]) -> list[float]:
    """
    Convert individual L2 distances to similarity scores.
    Used for per-chunk similarity display in the UI and logs.
    """
    return [1.0 / (1.0 + d) for d in distances] if distances else []
