"""
RAG Evaluation Pipeline
========================
Computes four quality metrics for each query-response pair:

1. Retrieval Quality  — Are we retrieving relevant documents?
2. Answer Relevance   — Does the answer address the question?
3. Context Utilization — Is the model using the retrieved context?
4. Faithfulness        — Is every claim supported by the context?

WHY evaluate every query:
Production ML systems require continuous evaluation, not just pre-deployment
benchmarks. Metrics drift as documents change, user queries evolve, and models
update. Continuous evaluation catches degradation before users do.

DESIGN DECISIONS:
- Retrieval quality and context utilization use fast heuristics (no LLM call).
- Answer relevance and faithfulness use LLM-as-judge (same model, zero extra cost).
- LLM-as-judge is the standard approach used by RAGAS, DeepEval, and TruLens.
- All functions are pure and stateless — the caller handles persistence.
"""

import re
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL
from observability.logger import update_log_evaluation


# Use a separate LLM instance for evaluation to avoid blocking the main pipeline
_eval_llm = ChatOllama(model=LLM_MODEL, temperature=0)


# ─── Metric 1: Retrieval Quality ───────────────────────────────────────

def compute_retrieval_quality(similarity_scores: list[float]) -> float:
    """
    Mean similarity score of retrieved documents.
    Higher = more relevant documents in the context window.

    This is the most fundamental RAG metric: if retrieval is bad,
    everything downstream (answer quality, faithfulness) will be bad.
    """
    if not similarity_scores:
        return 0.0
    return round(sum(similarity_scores) / len(similarity_scores), 4)


# ─── Metric 2: Answer Relevance (LLM-as-judge) ────────────────────────

_relevance_prompt = ChatPromptTemplate.from_template(
    "Rate how relevant the following answer is to the given question. "
    "Respond with ONLY a single number from 1 to 5.\n"
    "1 = Completely irrelevant\n"
    "2 = Mostly irrelevant\n"
    "3 = Partially relevant\n"
    "4 = Mostly relevant\n"
    "5 = Perfectly relevant\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Rating (1-5):"
)

_relevance_chain = _relevance_prompt | _eval_llm | StrOutputParser()


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Use LLM-as-judge to rate answer relevance on a 1-5 scale.
    Returns a normalized 0-1 score.
    """
    try:
        result = _relevance_chain.invoke({
            "question": question,
            "answer": answer,
        })
        # Extract the first number from the response
        match = re.search(r"[1-5]", result.strip())
        if match:
            return round((int(match.group()) - 1) / 4.0, 4)  # Normalize to 0-1
        return 0.5  # Default to mid-range if parsing fails
    except Exception:
        return 0.5


# ─── Metric 3: Context Utilization ─────────────────────────────────────

def compute_context_utilization(answer: str, context: str) -> float:
    """
    Token overlap ratio between the answer and retrieved context.

    Measures whether the model is actually using the retrieved information
    vs. relying on parametric knowledge. Low utilization suggests the model
    is ignoring the context (a common failure mode).

    This is a fast heuristic — no LLM call needed.
    """
    if not answer or not context:
        return 0.0

    # Tokenize by splitting on whitespace and lowering
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    # Remove common stopwords to focus on content words
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "don", "now", "and", "but", "or", "if", "it", "its", "this", "that",
    }

    answer_content = answer_tokens - stopwords
    context_content = context_tokens - stopwords

    if not answer_content:
        return 0.0

    overlap = answer_content & context_content
    return round(len(overlap) / len(answer_content), 4)


# ─── Metric 4: Faithfulness (LLM-as-judge) ─────────────────────────────

_faithfulness_prompt = ChatPromptTemplate.from_template(
    "Determine if EVERY claim in the Answer is supported by the Context. "
    "Respond with ONLY 'Yes' or 'No'.\n\n"
    "Context: {context}\n\n"
    "Answer: {answer}\n\n"
    "Is every claim in the answer supported by the context? (Yes/No):"
)

_faithfulness_chain = _faithfulness_prompt | _eval_llm | StrOutputParser()


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Use LLM-as-judge to determine if the answer is faithful to the context.
    Returns 1.0 (faithful) or 0.0 (unfaithful).

    Faithfulness is the most critical safety metric for RAG systems.
    An unfaithful answer means the model is hallucinating — generating
    claims not supported by the retrieved evidence.
    """
    if not answer or not context:
        return 0.0

    try:
        result = _faithfulness_chain.invoke({
            "context": context,
            "answer": answer,
        })
        cleaned = result.strip().lower()
        if "yes" in cleaned:
            return 1.0
        return 0.0
    except Exception:
        return 0.5  # Uncertain if evaluation fails


# ─── Async Evaluation Runner ───────────────────────────────────────────

async def evaluate_and_log(
    query_id: str,
    question: str,
    answer: str,
    context: str,
    similarity_scores: list[float],
) -> None:
    """
    Run all four evaluation metrics and update the log entry.

    This is designed to be called via asyncio.create_task() so it
    runs AFTER the response is returned to the user. The user never
    waits for evaluation to complete.
    """
    try:
        # Run heuristic metrics synchronously (fast)
        retrieval_quality = compute_retrieval_quality(similarity_scores)
        context_util = compute_context_utilization(answer, context)

        # Run LLM-based metrics (these take time but are non-blocking)
        loop = asyncio.get_event_loop()
        relevance = await loop.run_in_executor(
            None, compute_answer_relevance, question, answer
        )
        faithfulness = await loop.run_in_executor(
            None, compute_faithfulness, answer, context
        )

        # Update the log entry with evaluation results
        update_log_evaluation(
            query_id=query_id,
            retrieval_quality=retrieval_quality,
            answer_relevance=relevance,
            context_utilization=context_util,
            faithfulness=faithfulness,
        )
    except Exception as e:
        # Evaluation failure should never break the main pipeline
        print(f"[EVAL ERROR] Failed to evaluate query {query_id}: {e}")
