"""
Analytics Dashboard Router
============================
Exposes aggregate metrics and query logs for the frontend dashboard.

Endpoints:
- GET /analytics/summary   — Aggregate metrics (total queries, avg latency, etc.)
- GET /analytics/queries    — Paginated query log with eval data
- GET /analytics/feedback   — Feedback distribution
- GET /analytics/frequent   — Most frequently asked questions
"""

from typing import Optional

from fastapi import APIRouter, Query

from models import AnalyticsSummary, FrequentQuestion, QueryLogItem
from services.analytics_service import (
    get_analytics_summary,
    get_frequent_questions,
    get_query_logs,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary():
    """Get aggregate metrics across all logged queries."""
    return get_analytics_summary()


@router.get("/queries", response_model=list[QueryLogItem])
async def analytics_queries(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    feedback: Optional[str] = Query(None, pattern="^(positive|negative|none)$"),
    low_confidence: bool = Query(False),
):
    """
    Get paginated query logs with optional filtering.
    Supports filtering by feedback type and low-confidence flag.
    """
    return get_query_logs(
        limit=limit,
        offset=offset,
        feedback_filter=feedback,
        low_confidence_only=low_confidence,
    )


@router.get("/frequent", response_model=list[FrequentQuestion])
async def frequent_questions(
    top_n: int = Query(10, ge=1, le=50),
):
    """Get the most frequently asked questions."""
    return get_frequent_questions(top_n=top_n)
