"""
Analytics Aggregation Service
==============================
Queries the eval_logs ChromaDB collection and computes aggregate metrics
for the analytics dashboard.

WHY a separate service:
- Aggregation logic is decoupled from the API layer (routers).
- Can be unit tested with mock data.
- Could be swapped to query a different backend (e.g., SQL) without
  changing the router interface.

NOTE ON ChromaDB LIMITATIONS:
ChromaDB is a vector database, not a SQL database. It doesn't support
GROUP BY, AVG(), or COUNT() natively. We fetch all records and compute
aggregations in Python. For a production system with millions of logs,
you'd want a proper analytics database (ClickHouse, BigQuery, etc.).
For this scale, in-memory aggregation is fast enough.
"""

import json
from collections import Counter
from typing import Optional

from db.collections import eval_logs_collection
from models import AnalyticsSummary, FrequentQuestion, QueryLogItem
from config import CONFIDENCE_THRESHOLD


def get_all_logs() -> list[dict]:
    """Fetch all entries from eval_logs collection."""
    result = eval_logs_collection.get(
        include=["metadatas", "documents"],
    )

    logs = []
    for i, query_id in enumerate(result["ids"]):
        meta = result["metadatas"][i]
        meta["query_id"] = query_id
        logs.append(meta)

    return logs


def get_analytics_summary() -> AnalyticsSummary:
    """Compute aggregate metrics across all logged queries."""
    logs = get_all_logs()

    if not logs:
        return AnalyticsSummary(
            total_queries=0,
            avg_response_latency_ms=0.0,
            avg_similarity_score=0.0,
            positive_feedback=0,
            negative_feedback=0,
            no_feedback=0,
            failed_retrievals=0,
            low_confidence_queries=0,
        )

    total = len(logs)
    latencies = [log.get("response_latency_ms", 0) for log in logs]
    confidences = [log.get("confidence_score", 0) for log in logs]

    # Feedback counts
    feedbacks = [log.get("feedback", "none") for log in logs]
    positive = feedbacks.count("positive")
    negative = feedbacks.count("negative")
    no_feedback = total - positive - negative

    # Failed retrievals: all similarity scores below threshold
    failed = sum(
        1 for log in logs
        if log.get("confidence_score", 0) < CONFIDENCE_THRESHOLD * 0.5
    )

    # Low confidence queries
    low_conf = sum(
        1 for log in logs
        if log.get("is_low_confidence", False)
    )

    # Evaluation metric averages (only from logs that have them)
    def avg_metric(key: str) -> Optional[float]:
        values = [log[key] for log in logs if key in log and log[key] is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return AnalyticsSummary(
        total_queries=total,
        avg_response_latency_ms=round(sum(latencies) / total, 2) if total else 0.0,
        avg_similarity_score=round(sum(confidences) / total, 4) if total else 0.0,
        positive_feedback=positive,
        negative_feedback=negative,
        no_feedback=no_feedback,
        failed_retrievals=failed,
        low_confidence_queries=low_conf,
        avg_retrieval_quality=avg_metric("retrieval_quality"),
        avg_answer_relevance=avg_metric("answer_relevance"),
        avg_context_utilization=avg_metric("context_utilization"),
        avg_faithfulness=avg_metric("faithfulness"),
    )


def get_frequent_questions(top_n: int = 10) -> list[FrequentQuestion]:
    """
    Get the most frequently asked questions.
    Uses exact string matching — for production, you'd want semantic
    deduplication (cluster similar questions by embedding distance).
    """
    logs = get_all_logs()
    questions = [log.get("user_question", "") for log in logs]
    counter = Counter(questions)

    return [
        FrequentQuestion(question=q, count=c)
        for q, c in counter.most_common(top_n)
        if q  # Skip empty questions
    ]


def get_query_logs(
    limit: int = 50,
    offset: int = 0,
    feedback_filter: Optional[str] = None,
    low_confidence_only: bool = False,
) -> list[QueryLogItem]:
    """
    Get paginated query logs with optional filtering.
    Supports filtering by feedback type and low-confidence queries.
    """
    logs = get_all_logs()

    # Apply filters
    if feedback_filter:
        logs = [l for l in logs if l.get("feedback") == feedback_filter]

    if low_confidence_only:
        logs = [l for l in logs if l.get("is_low_confidence", False)]

    # Sort by timestamp (newest first)
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Paginate
    paginated = logs[offset: offset + limit]

    return [
        QueryLogItem(
            query_id=log.get("query_id", ""),
            user_question=log.get("user_question", ""),
            generated_answer=log.get("generated_answer", ""),
            confidence_score=log.get("confidence_score", 0.0),
            response_latency_ms=log.get("response_latency_ms", 0.0),
            feedback=log.get("feedback") if log.get("feedback") != "none" else None,
            timestamp=log.get("timestamp", ""),
            retrieval_quality=log.get("retrieval_quality"),
            faithfulness=log.get("faithfulness"),
        )
        for log in paginated
    ]
