# 🧠 RAG Studio — Evaluation & Observability Platform

<p align="center">
  <em>A production-oriented PDF Question-Answering system with built-in evaluation pipelines, observability, and analytics — designed to demonstrate MLOps engineering maturity.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=google-chrome&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
</p>

---

## 📌 What Makes This Different?

Most RAG projects stop at "upload PDF → ask question → get answer." This project goes further by adding the **production infrastructure** that separates a student project from a real-world AI system:

| Student RAG | This Project |
|-------------|-------------|
| Single-file backend | Modular architecture (routers, services, observability, db) |
| No quality monitoring | 4 automated evaluation metrics on every query |
| Blind to failures | Confidence scoring with circuit breaker pattern |
| No audit trail | Structured logging with trace IDs for every request |
| No user feedback | 👍/👎 feedback system tied to evaluation data |
| No visibility | Real-time analytics dashboard |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         React Frontend (Vite)                        │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  Chat Page   │  │  Feedback (👍👎) │  │  Analytics Dashboard │   │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬───────────┘   │
└─────────┼──────────────────┼────────────────────────┼───────────────┘
          │                  │                        │
          ▼                  ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                                 │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  Upload   │  │    Chat    │  │ Feedback │  │    Analytics      │  │
│  │  Router   │  │   Router   │  │  Router  │  │     Router        │  │
│  └────┬─────┘  └──────┬─────┘  └────┬─────┘  └─────────┬─────────┘  │
│       │               │             │                   │            │
│  ┌────┴───────────────┴─────────────┴───────────────────┴─────────┐  │
│  │                     Services Layer                              │  │
│  │  RAG Pipeline │ Confidence │ Evaluation │ Analytics Service     │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                          │
│  ┌────────────────────────┴───────────────────────────────────────┐  │
│  │                   Observability Layer                           │  │
│  │              Logger │ Metrics │ Structured Logging              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │       ChromaDB          │
              │  ┌─────────┐ ┌────────┐ │
              │  │Documents│ │Eval    │ │
              │  │Collection│ │Logs   │ │
              │  └─────────┘ └────────┘ │
              └─────────────────────────┘
```

---

## ✨ Features

### 🔍 Core RAG Pipeline
- **PDF Upload** — Drag-and-drop PDF ingestion with automatic text extraction, recursive character chunking (1000 chars / 200 overlap), and vector storage
- **Context-Aware Chat** — Multi-turn conversations using query reformulation with chat history
- **Persistent Storage** — ChromaDB embeddings stored to disk, survive server restarts
- **Source Attribution** — Every answer shows the retrieved document chunks used

### 📊 Evaluation Pipeline (4 Automated Metrics)
Every query is automatically evaluated on four quality dimensions:

| Metric | Method | What It Catches |
|--------|--------|-----------------|
| **Retrieval Quality** | Mean similarity score | Poor embeddings, wrong chunk sizes, irrelevant retrievals |
| **Answer Relevance** | LLM-as-judge (1–5 rating → normalized 0–1) | Off-topic or vague answers |
| **Context Utilization** | Token overlap ratio (answer vs. context) | Model ignoring retrieved context, relying on parametric knowledge |
| **Faithfulness** | LLM-as-judge (Yes/No binary) | Hallucinated claims not supported by context |

> Evaluation runs **asynchronously** after the response is sent — zero impact on user-perceived latency.

### 🛡️ Confidence Scoring & Circuit Breaker
- Converts ChromaDB L2 distances to a **0–1 confidence score** using `1 / (1 + mean(distances))`
- If confidence falls below the configurable threshold (default: `0.35`), the system **skips the LLM call entirely** and returns:
  > *"I couldn't find sufficient relevant information in the provided documents."*
- This prevents hallucination **and** saves compute on hopeless queries

### 📝 Structured Observability Logging
Every query generates a structured log entry with **14+ fields**:

| Field | Description |
|-------|-------------|
| `query_id` | Unique UUID trace ID |
| `user_question` | Original user input |
| `standalone_query` | Reformulated standalone query |
| `retrieved_chunks` | Document chunks used as context |
| `similarity_scores` | Per-chunk similarity scores |
| `retrieved_doc_ids` | Source document + page references |
| `generated_answer` | LLM output |
| `response_latency_ms` | End-to-end pipeline latency |
| `estimated_tokens` | Token count estimate (chars/4 heuristic) |
| `confidence_score` | Retrieval confidence (0–1) |
| `is_low_confidence` | Whether circuit breaker triggered |
| `timestamp` | ISO 8601 UTC timestamp |
| `feedback` | User feedback (positive/negative/none) |
| `evaluation metrics` | All 4 eval scores (populated async) |

Logs are stored in a **separate ChromaDB collection** (`eval_logs`) — isolated from document embeddings to avoid polluting retrieval results.

### 👍👎 User Feedback System
- Thumbs up/down buttons on every assistant response
- Feedback stored alongside evaluation data for **correlation analysis**
- Enables questions like: *"Do queries with low faithfulness also get negative feedback?"*

### 📈 Analytics Dashboard
Real-time dashboard showing:
- **Key Metrics** — Total queries, average latency, average confidence, low-confidence count
- **Evaluation Quality Bars** — Visual progress bars for all 4 eval metrics
- **Feedback Distribution** — Stacked bar showing positive / negative / no-feedback ratio
- **Most Frequent Questions** — Ranked list with query counts
- **Query Log Table** — Sortable table with confidence pills, faithfulness status, and timestamps
- All visualizations built with **pure CSS** — no chart library dependencies

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| [Python](https://www.python.org/) | 3.10+ | Backend runtime |
| [Node.js](https://nodejs.org/) | 16+ | Frontend build tooling |
| [Ollama](https://ollama.com/) | Latest | Local LLM inference |

### Step 1: Pull Ollama Models

```bash
ollama pull nomic-embed-text    # Embedding model
ollama pull llama3.2            # Chat/generation model
```

Make sure Ollama is running:
```bash
ollama serve
```

### Step 2: Start the Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

- **API**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Step 3: Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

- **UI**: http://localhost:5173

### Step 4: Use the Application

1. Open http://localhost:5173
2. Upload a PDF using the drag-and-drop uploader
3. Ask questions about the document content
4. Observe confidence scores and evaluation metrics on each response
5. Provide 👍/👎 feedback
6. Switch to the **Dashboard** tab to view analytics

---

## 📁 Project Structure

```
backend/
├── main.py                     # FastAPI app setup, CORS, router mounting, error handling
├── config.py                   # Centralized configuration (thresholds, model names, etc.)
├── models.py                   # Pydantic request/response schemas
├── memory.py                   # In-memory session chat history
├── requirements.txt            # Python dependencies
│
├── routers/                    # API endpoint layer
│   ├── upload.py               # POST /upload — PDF ingestion
│   ├── chat.py                 # POST /chat — Query with confidence gating + async eval
│   ├── feedback.py             # POST /feedback — User feedback submission
│   └── analytics.py            # GET /analytics/* — Dashboard data endpoints
│
├── services/                   # Business logic layer
│   ├── rag_pipeline.py         # Core RAG: reformulate → retrieve (with scores) → generate
│   ├── confidence.py           # Confidence scoring & circuit breaker threshold gating
│   ├── evaluation.py           # 4-metric evaluation pipeline (async, LLM-as-judge)
│   └── analytics_service.py    # Aggregation queries for dashboard metrics
│
├── observability/              # Monitoring & logging layer
│   ├── logger.py               # Structured query logging to ChromaDB eval_logs collection
│   └── metrics.py              # Token estimation, confidence computation, distance conversion
│
└── db/
    └── collections.py          # ChromaDB collection management (documents + eval_logs)

frontend/
├── index.html                  # Entry point with SEO meta tags
├── vite.config.js              # Vite configuration
├── package.json                # Node dependencies
│
└── src/
    ├── main.jsx                # React DOM render
    ├── App.jsx                 # Root component with sidebar page routing
    ├── App.css                 # Layout styles
    ├── index.css               # Design system (dark theme, glassmorphism, animations)
    │
    ├── api/
    │   └── client.js           # Centralized API wrapper (chat, upload, feedback, analytics)
    │
    ├── hooks/
    │   └── useChat.js          # Chat state management (messages, loading, feedback, sessions)
    │
    ├── components/
    │   ├── ChatInterface.jsx   # Message list, input bar, typing indicator, empty state
    │   ├── ChatMessage.jsx     # Message bubble with feedback buttons & source expansion
    │   ├── ConfidenceIndicator.jsx  # Color-coded confidence badge with tooltip
    │   ├── PdfUploader.jsx     # Drag-and-drop PDF upload with progress states
    │   └── Sidebar.jsx         # Navigation sidebar with branding
    │
    └── pages/
        ├── ChatPage.jsx        # Chat + collapsible upload panel
        └── DashboardPage.jsx   # Analytics dashboard (metric cards, charts, query log)
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{ status: "healthy" }` |
| `POST` | `/upload` | Upload a PDF file for processing |
| `POST` | `/chat` | Send a query — returns answer, confidence, sources, query_id |
| `POST` | `/feedback` | Submit 👍/👎 feedback for a query_id |
| `GET` | `/analytics/summary` | Aggregate metrics (total queries, avg latency, etc.) |
| `GET` | `/analytics/queries` | Paginated query logs with eval data |
| `GET` | `/analytics/frequent` | Most frequently asked questions |

Full interactive documentation available at http://localhost:8000/docs after starting the backend.

---

## ⚙️ Configuration

All tunable parameters are centralized in [`backend/config.py`](backend/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.35` | Below this → circuit breaker activates, LLM is skipped |
| `RETRIEVER_TOP_K` | `3` | Number of document chunks retrieved per query |
| `LLM_MODEL` | `llama3.2` | Ollama model for generation and evaluation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `LLM_TEMPERATURE` | `0` | Generation temperature (0 = deterministic) |
| `EVAL_ASYNC` | `True` | Run evaluation asynchronously (non-blocking) |
| `EVAL_SAMPLE_RATE` | `1.0` | Fraction of queries to evaluate (1.0 = all) |
| `CHARS_PER_TOKEN` | `4` | Token estimation heuristic for English text |

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19 + Vite | UI framework and build tooling |
| **Backend** | FastAPI + Uvicorn | Async API server |
| **LLM** | Ollama (Llama 3.2) | Local inference — no API keys needed |
| **Embeddings** | nomic-embed-text | Document embedding via Ollama |
| **Vector Store** | ChromaDB | Persistent vector storage + eval log storage |
| **Orchestration** | LangChain | RAG pipeline (loaders, splitters, chains, prompts) |
| **Styling** | Vanilla CSS | Dark theme with glassmorphism — no CSS framework |

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ConnectionError: Failed to connect to Ollama` | Ollama is not running | Run `ollama serve` in a separate terminal |
| `CORS policy: No 'Access-Control-Allow-Origin'` | Server 500 error masking as CORS | Check terminal logs for the actual error (usually Ollama not running) |
| `model not found` | Ollama model not pulled | Run `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| Dashboard shows "No Data Yet" | No queries have been made yet | Upload a PDF and ask questions first |
| Slow responses | First query warms up the model | Subsequent queries will be faster — Ollama caches the model in memory |

---

## 🔄 MLOps Principles Demonstrated

This project implements core MLOps practices that are critical for production AI systems:

| MLOps Principle | How It's Implemented | Why It Matters |
|----------------|---------------------|----------------|
| **Monitoring** | Every query logged with latency, confidence, and 4 eval scores | Detect degradation before users complain |
| **Observability** | Trace IDs (`query_id`) enable end-to-end debugging | Investigate any individual query's full pipeline |
| **Continuous Evaluation** | Automated metrics run on every query (async) | Catches quality drift as documents/queries change |
| **Feedback Loops** | User feedback (👍/👎) correlated with automated metrics | Ground truth for validating automated scores |
| **Circuit Breakers** | Confidence threshold prevents low-quality answers | Protect users from hallucination; save compute |
| **Separation of Concerns** | Routers → Services → Observability → DB layers | Independently testable, deployable, replaceable |
| **Data Flywheel** | Eval logs + feedback = labeled dataset | Foundation for fine-tuning and retraining |
| **Configuration Management** | All thresholds in `config.py`, not hardcoded | Reproducible, auditable, version-controlled |
| **Graceful Degradation** | Evaluation failures never break the main pipeline | Production resilience — observability can't cause outages |
| **API Documentation** | Auto-generated OpenAPI docs at `/docs` | Self-documenting API from Pydantic schemas |

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
