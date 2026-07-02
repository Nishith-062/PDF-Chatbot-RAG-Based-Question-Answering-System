"""
Confidence Scoring Service
===========================
Implements the confidence threshold gating pattern — a production
reliability mechanism that prevents the system from serving low-quality
answers when retrieval confidence is poor.

WHY this is critical:
Without confidence gating, a RAG system will generate authoritative-
sounding answers from irrelevant context. This is the #1 production
failure mode for RAG systems. The threshold acts as a CIRCUIT BREAKER
— a core MLOps pattern borrowed from microservices architecture.

HOW it works:
1. Compute confidence from retrieval distances
2. If below threshold → return safe fallback message (no LLM call)
3. If above threshold → proceed with normal generation

This saves compute on hopeless queries AND protects user trust.
"""

from config import CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_RESPONSE
from observability.metrics import compute_confidence


def check_confidence(distances: list[float]) -> tuple[float, bool, str | None]:
    """
    Evaluate retrieval confidence and decide whether to proceed with generation.

    Args:
        distances: L2 distances from ChromaDB retrieval

    Returns:
        (confidence_score, is_low_confidence, fallback_response_or_none)
        - If is_low_confidence is True, fallback_response contains the safe message
        - If is_low_confidence is False, fallback_response is None (proceed normally)
    """
    confidence = compute_confidence(distances)
    is_low = confidence < CONFIDENCE_THRESHOLD

    if is_low:
        return confidence, True, LOW_CONFIDENCE_RESPONSE
    return confidence, False, None
