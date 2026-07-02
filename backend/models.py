"""
Pydantic Models (Request / Response Schemas)
=============================================
Centralizing all schemas improves API documentation (auto-generated
by FastAPI/Swagger) and enables type-safe development across the codebase.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ─── Chat ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    query_id: str = Field(..., description="Unique trace ID for this query")
    answer: str
    sources: List[str]
    standalone_query: str
    confidence_score: float = Field(..., description="0-1 confidence in retrieval quality")
    is_low_confidence: bool = Field(False, description="True if confidence below threshold")


# ─── Feedback ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str = Field(..., pattern="^(positive|negative)$")


class FeedbackResponse(BaseModel):
    status: str
    query_id: str


# ─── Evaluation Log Entry ──────────────────────────────────────────────

class EvalLogEntry(BaseModel):
    query_id: str
    session_id: str
    user_question: str
    standalone_query: str
    retrieved_chunks: List[str]
    similarity_scores: List[float]
    retrieved_doc_ids: List[str]
    generated_answer: str
    response_latency_ms: float
    estimated_tokens: int
    confidence_score: float
    is_low_confidence: bool
    timestamp: str
    feedback: Optional[str] = None

    # Evaluation metrics (populated asynchronously)
    retrieval_quality: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_utilization: Optional[float] = None
    faithfulness: Optional[float] = None


# ─── Analytics ──────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_queries: int
    avg_response_latency_ms: float
    avg_similarity_score: float
    positive_feedback: int
    negative_feedback: int
    no_feedback: int
    failed_retrievals: int
    low_confidence_queries: int
    avg_retrieval_quality: Optional[float] = None
    avg_answer_relevance: Optional[float] = None
    avg_context_utilization: Optional[float] = None
    avg_faithfulness: Optional[float] = None


class FrequentQuestion(BaseModel):
    question: str
    count: int


class QueryLogItem(BaseModel):
    query_id: str
    user_question: str
    generated_answer: str
    confidence_score: float
    response_latency_ms: float
    feedback: Optional[str] = None
    timestamp: str
    retrieval_quality: Optional[float] = None
    faithfulness: Optional[float] = None
