"""
Centralized Configuration
=========================
All tunable parameters in one place. In production, these would be loaded
from environment variables or a config service. Centralizing them here
follows the MLOps principle of "configuration as code" — every threshold
is versioned, auditable, and reproducible.
"""

# ─── Model Configuration ───────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0

# ─── ChromaDB Collections ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
DOCUMENT_COLLECTION = "langchain"          # Default LangChain collection name
EVAL_LOGS_COLLECTION = "eval_logs"         # Separate collection for observability

# ─── Retrieval Settings ────────────────────────────────────────────────
RETRIEVER_TOP_K = 3

# ─── Confidence & Threshold Gating ─────────────────────────────────────
# Confidence score below this threshold triggers a safe "I don't know" response
# instead of risking hallucination. This acts as a circuit breaker.
CONFIDENCE_THRESHOLD = 0.35

LOW_CONFIDENCE_RESPONSE = (
    "I couldn't find sufficient relevant information in the provided documents. "
    "Please try rephrasing your question or upload a more relevant document."
)

# ─── Evaluation Settings ───────────────────────────────────────────────
EVAL_ASYNC = True    # Run evaluation metrics asynchronously (non-blocking)
EVAL_SAMPLE_RATE = 1.0  # Fraction of queries to evaluate (1.0 = all)

# ─── Token Estimation ──────────────────────────────────────────────────
# Ollama doesn't expose token counts. This is a standard heuristic for English text.
CHARS_PER_TOKEN = 4
