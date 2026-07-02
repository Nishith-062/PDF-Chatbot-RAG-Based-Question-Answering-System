"""
ChromaDB Collection Management
===============================
Singleton-style module that initializes and exposes both the document
vector store (used by the RAG pipeline) and the evaluation logs collection
(used by the observability layer).

WHY separate collections:
- The documents collection stores PDF chunk embeddings for retrieval.
- The eval_logs collection stores structured metadata about each query.
  Mixing them would pollute retrieval results — a retrieved "eval log"
  is not a document the user uploaded.
- This follows the data-plane / control-plane separation pattern used
  in production ML systems.
"""

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    EVAL_LOGS_COLLECTION,
    RETRIEVER_TOP_K,
)

# ─── Shared Embedding Function ─────────────────────────────────────────
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# ─── Document Vector Store (existing RAG pipeline) ─────────────────────
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

# ─── Evaluation Logs Collection (observability) ────────────────────────
# Uses raw ChromaDB client for structured metadata storage.
# We don't need LangChain's Chroma wrapper here because we're not doing
# similarity search on eval logs — just CRUD operations.
_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
eval_logs_collection = _chroma_client.get_or_create_collection(
    name=EVAL_LOGS_COLLECTION
)
