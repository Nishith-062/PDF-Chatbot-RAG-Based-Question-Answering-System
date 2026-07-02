"""
Feedback Router
================
Handles user feedback (👍/👎) for individual query responses.

WHY feedback matters:
User feedback is the ONLY ground truth signal in a production RAG system.
Automated metrics (retrieval quality, faithfulness) are proxies — real user
satisfaction is what we ultimately optimize for.

Storing feedback alongside eval metrics enables CORRELATION ANALYSIS:
- "Do queries with low faithfulness scores also get negative feedback?"
- "What confidence threshold maximizes positive feedback?"
- This data is the foundation for the MLOps continuous improvement loop.
"""

from fastapi import APIRouter, HTTPException

from models import FeedbackRequest, FeedbackResponse
from observability.logger import update_log_feedback

router = APIRouter(tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a specific query.
    Updates the eval_logs entry with the feedback value.
    """
    success = update_log_feedback(request.query_id, request.feedback)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Query log entry not found for query_id: {request.query_id}",
        )

    return FeedbackResponse(
        status="feedback_recorded",
        query_id=request.query_id,
    )
