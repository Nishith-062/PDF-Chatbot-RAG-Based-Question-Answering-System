"""
Chat Router
============
Handles the chat endpoint with integrated:
- Confidence scoring & threshold gating
- Structured logging (observability)
- Async evaluation triggering

FLOW:
1. Receive user query
2. Reformulate query (if chat history exists)
3. Retrieve documents with similarity scores
4. Check confidence → if low, return safe response (skip LLM)
5. Generate answer from context
6. Log structured entry to eval_logs collection
7. Trigger async evaluation (non-blocking)
8. Return response with confidence metadata
"""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter

from models import ChatRequest, ChatResponse
from services.rag_pipeline import (
    run_pipeline,
    run_pipeline_without_generation,
    RetrievalResult,
)
from services.confidence import check_confidence
from services.evaluation import evaluate_and_log
from observability.logger import generate_query_id, log_query
from observability.metrics import estimate_tokens
from memory import memory_store
from models import EvalLogEntry
from config import LOW_CONFIDENCE_RESPONSE, EVAL_ASYNC

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query through the RAG pipeline with confidence gating,
    structured logging, and async evaluation.
    """
    query_id = generate_query_id()
    session_id = request.session_id
    user_query = request.query

    # ── Step 1-2: Reformulate and Retrieve ──────────────────────────
    # First, run retrieval to get confidence before deciding whether
    # to invoke the (expensive) LLM generation step.
    pipeline_result = run_pipeline_without_generation(user_query, session_id)

    # ── Step 3: Confidence Check ────────────────────────────────────
    confidence, is_low_confidence, fallback = check_confidence(
        pipeline_result.retrieval.distances
    )

    if is_low_confidence:
        # Circuit breaker activated — skip LLM, return safe response
        answer = LOW_CONFIDENCE_RESPONSE
        latency_ms = pipeline_result.latency_ms
    else:
        # Full pipeline with generation
        pipeline_result = run_pipeline(user_query, session_id)
        answer = pipeline_result.answer
        latency_ms = pipeline_result.latency_ms

    # ── Step 4: Update Chat Memory ──────────────────────────────────
    memory_store.add_message(session_id, "user", user_query)
    memory_store.add_message(session_id, "assistant", answer)

    # ── Step 5: Structured Logging ──────────────────────────────────
    total_text = (
        user_query
        + pipeline_result.standalone_query
        + "\n\n".join(pipeline_result.retrieval.chunks)
        + answer
    )

    log_entry = EvalLogEntry(
        query_id=query_id,
        session_id=session_id,
        user_question=user_query,
        standalone_query=pipeline_result.standalone_query,
        retrieved_chunks=pipeline_result.retrieval.chunks,
        similarity_scores=pipeline_result.retrieval.similarities,
        retrieved_doc_ids=pipeline_result.retrieval.doc_ids,
        generated_answer=answer,
        response_latency_ms=latency_ms,
        estimated_tokens=estimate_tokens(total_text),
        confidence_score=round(confidence, 4),
        is_low_confidence=is_low_confidence,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    log_query(log_entry)

    # ── Step 6: Async Evaluation ────────────────────────────────────
    if EVAL_ASYNC and not is_low_confidence and pipeline_result.retrieval.chunks:
        context = "\n\n".join(pipeline_result.retrieval.chunks)
        asyncio.create_task(
            evaluate_and_log(
                query_id=query_id,
                question=pipeline_result.standalone_query,
                answer=answer,
                context=context,
                similarity_scores=pipeline_result.retrieval.similarities,
            )
        )

    # ── Step 7: Return Response ─────────────────────────────────────
    return ChatResponse(
        query_id=query_id,
        answer=answer,
        sources=pipeline_result.retrieval.chunks,
        standalone_query=pipeline_result.standalone_query,
        confidence_score=round(confidence, 4),
        is_low_confidence=is_low_confidence,
    )
