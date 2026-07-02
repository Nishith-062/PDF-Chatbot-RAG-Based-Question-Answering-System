"""
Core RAG Pipeline Service
==========================
Extracted from the original main.py. Contains the three-stage RAG pipeline:
1. Query Reformulation (contextual using chat history)
2. Retrieval (ChromaDB similarity search with raw distances)
3. Generation (LLM answer synthesis)

WHY extract this:
- The original main.py mixed HTTP handling with ML pipeline logic.
- Extracting the pipeline makes it testable independently of FastAPI.
- Each stage can be instrumented, cached, or replaced without touching
  the API layer — a key MLOps principle (loose coupling).

WHAT CHANGED from the original:
- Retriever now returns raw distances (for confidence scoring)
- Pipeline returns structured data (for logging and evaluation)
- No behavioral changes to the prompt templates or chain logic
"""

import time
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K
from db.collections import vectorstore, retriever
from memory import memory_store
from observability.metrics import distances_to_similarities, estimate_tokens


@dataclass
class RetrievalResult:
    """Structured output from the retrieval stage."""
    chunks: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete output from the RAG pipeline."""
    standalone_query: str = ""
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    answer: str = ""
    latency_ms: float = 0.0
    estimated_tokens: int = 0


# ─── LLM Instance ──────────────────────────────────────────────────────
llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# ─── Stage 1: Query Reformulation ──────────────────────────────────────

_reformulate_prompt = ChatPromptTemplate.from_template(
    "Given the following chat history and a new user question, "
    "rephrase the new question to be a standalone question that can be "
    "understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "New Question: {question}\n\n"
    "Standalone Question:"
)

_reformulate_chain = _reformulate_prompt | llm | StrOutputParser()


def reformulate_query(user_query: str, session_id: str) -> str:
    """
    If there's chat history, reformulate the query to be standalone.
    Otherwise, return the query as-is.
    """
    chat_history = memory_store.get_history_string(session_id)
    if chat_history.strip():
        return _reformulate_chain.invoke({
            "chat_history": chat_history,
            "question": user_query,
        })
    return user_query


# ─── Stage 2: Retrieval ────────────────────────────────────────────────

def retrieve_with_scores(query: str) -> RetrievalResult:
    """
    Retrieve documents with raw similarity distances.

    Uses ChromaDB's native similarity_search_with_score instead of
    LangChain's retriever to get access to distance values needed
    for confidence scoring.
    """
    results = vectorstore.similarity_search_with_score(query, k=RETRIEVER_TOP_K)

    if not results:
        return RetrievalResult()

    chunks = []
    doc_ids = []
    distances = []

    for doc, distance in results:
        chunks.append(doc.page_content)
        # Use metadata source + page as doc ID, fallback to chunk hash
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        doc_ids.append(f"{source}:page_{page}")
        distances.append(float(distance))

    return RetrievalResult(
        chunks=chunks,
        doc_ids=doc_ids,
        distances=distances,
        similarities=distances_to_similarities(distances),
    )


# ─── Stage 3: Generation ───────────────────────────────────────────────

_qa_prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_qa_chain = _qa_prompt | llm | StrOutputParser()


def generate_answer(query: str, context: str) -> str:
    """Generate an answer using the LLM with retrieved context."""
    return _qa_chain.invoke({
        "context": context,
        "question": query,
    })


# ─── Full Pipeline ─────────────────────────────────────────────────────

def run_pipeline(user_query: str, session_id: str) -> PipelineResult:
    """
    Execute the complete RAG pipeline:
    1. Reformulate query (if chat history exists)
    2. Retrieve relevant documents with scores
    3. (Confidence check happens in the router, not here)
    4. Generate answer from context

    This function is called by the chat router, which handles confidence
    gating before invoking generation.
    """
    start = time.perf_counter()

    # Stage 1: Reformulation
    standalone_query = reformulate_query(user_query, session_id)

    # Stage 2: Retrieval
    retrieval = retrieve_with_scores(standalone_query)

    # Build context string
    context = "\n\n".join(retrieval.chunks) if retrieval.chunks else ""

    # Stage 3: Generation (may be skipped by confidence gating in router)
    answer = generate_answer(standalone_query, context) if context else ""

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Estimate tokens across all text
    total_text = user_query + standalone_query + context + answer
    tokens = estimate_tokens(total_text)

    return PipelineResult(
        standalone_query=standalone_query,
        retrieval=retrieval,
        answer=answer,
        latency_ms=round(elapsed_ms, 2),
        estimated_tokens=tokens,
    )


def run_pipeline_without_generation(
    user_query: str, session_id: str
) -> PipelineResult:
    """
    Run only reformulation and retrieval stages (skip generation).
    Used when confidence gating determines the query should be rejected.
    This saves LLM compute on low-confidence queries.
    """
    start = time.perf_counter()

    standalone_query = reformulate_query(user_query, session_id)
    retrieval = retrieve_with_scores(standalone_query)

    elapsed_ms = (time.perf_counter() - start) * 1000

    total_text = user_query + standalone_query + "\n\n".join(retrieval.chunks)
    tokens = estimate_tokens(total_text)

    return PipelineResult(
        standalone_query=standalone_query,
        retrieval=retrieval,
        answer="",  # No answer generated
        latency_ms=round(elapsed_ms, 2),
        estimated_tokens=tokens,
    )
