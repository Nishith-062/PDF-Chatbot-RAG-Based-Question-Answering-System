"""
Structured Query Logger
========================
Writes a structured log entry to the ChromaDB `eval_logs` collection
for every query processed by the RAG pipeline.

WHY structured logging matters:
- Every production ML system needs an audit trail. If a user reports
  a bad answer, you need to trace: what was the query? What context
  was retrieved? What was the confidence? What did the model output?
- Structured logs (vs. plain text logs) enable programmatic analysis:
  aggregation, filtering, trend detection, and dashboard visualization.
- Storing logs in a database (vs. log files) enables real-time querying
  without log parsing infrastructure.

DESIGN DECISION — ChromaDB for eval logs:
ChromaDB is not ideal for structured metadata (SQL would be better),
but using it keeps the stack simple and avoids adding another dependency.
The eval_logs collection uses ChromaDB as a document store with metadata,
not as a vector store — we don't embed the logs.
"""

import json
import uuid
from datetime import datetime, timezone

from db.collections import eval_logs_collection
from models import EvalLogEntry


def generate_query_id() -> str:
    """Generate a unique trace ID for each query."""
    return str(uuid.uuid4())


def log_query(entry: EvalLogEntry) -> None:
    """
    Persist a query log entry to the eval_logs ChromaDB collection.

    Each entry is stored as:
    - document: the user question (required by ChromaDB)
    - id: the query_id (unique trace ID)
    - metadata: all structured fields (flattened for ChromaDB compatibility)
    """
    # ChromaDB metadata values must be str, int, float, or bool.
    # Lists must be serialized to JSON strings.
    metadata = {
        "session_id": entry.session_id,
        "user_question": entry.user_question,
        "standalone_query": entry.standalone_query,
        "retrieved_chunks": json.dumps(entry.retrieved_chunks),
        "similarity_scores": json.dumps(entry.similarity_scores),
        "retrieved_doc_ids": json.dumps(entry.retrieved_doc_ids),
        "generated_answer": entry.generated_answer,
        "response_latency_ms": entry.response_latency_ms,
        "estimated_tokens": entry.estimated_tokens,
        "confidence_score": entry.confidence_score,
        "is_low_confidence": entry.is_low_confidence,
        "timestamp": entry.timestamp,
        "feedback": entry.feedback or "none",
    }

    # Add evaluation metrics if available
    if entry.retrieval_quality is not None:
        metadata["retrieval_quality"] = entry.retrieval_quality
    if entry.answer_relevance is not None:
        metadata["answer_relevance"] = entry.answer_relevance
    if entry.context_utilization is not None:
        metadata["context_utilization"] = entry.context_utilization
    if entry.faithfulness is not None:
        metadata["faithfulness"] = entry.faithfulness

    eval_logs_collection.upsert(
        ids=[entry.query_id],
        documents=[entry.user_question],
        metadatas=[metadata],
    )


def update_log_feedback(query_id: str, feedback: str) -> bool:
    """
    Update the feedback field on an existing log entry.
    Returns True if the entry was found and updated.
    """
    try:
        result = eval_logs_collection.get(ids=[query_id])
        if not result["ids"]:
            return False

        existing_metadata = result["metadatas"][0]
        existing_metadata["feedback"] = feedback
        eval_logs_collection.update(
            ids=[query_id],
            metadatas=[existing_metadata],
        )
        return True
    except Exception:
        return False


def update_log_evaluation(
    query_id: str,
    retrieval_quality: float,
    answer_relevance: float,
    context_utilization: float,
    faithfulness: float,
) -> bool:
    """
    Update evaluation metrics on an existing log entry.
    Called asynchronously after the response is returned to the user.
    """
    try:
        result = eval_logs_collection.get(ids=[query_id])
        if not result["ids"]:
            return False

        existing_metadata = result["metadatas"][0]
        existing_metadata["retrieval_quality"] = retrieval_quality
        existing_metadata["answer_relevance"] = answer_relevance
        existing_metadata["context_utilization"] = context_utilization
        existing_metadata["faithfulness"] = faithfulness
        eval_logs_collection.update(
            ids=[query_id],
            metadatas=[existing_metadata],
        )
        return True
    except Exception:
        return False
