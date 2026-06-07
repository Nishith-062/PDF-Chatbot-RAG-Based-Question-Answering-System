# Retrieval-Augmented Generation (RAG) Web Application

This is a full-stack Retrieval-Augmented Generation (RAG) web application. It allows users to upload PDF documents and ask questions about the content of those documents using a Large Language Model (LLM) through an intuitive chat interface.

## Project Structure

- `backend/`: FastAPI backend server handling document ingestion, chunking, vector storage (ChromaDB), and the LangChain query pipeline.
- `frontend/`: React frontend application built with Vite, providing the user interface for PDF uploads and chatting.

## Prerequisites

Before you begin, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (v16 or higher recommended)
- [Python](https://www.python.org/) (3.9 or higher)
- [Ollama](https://ollama.com/) (Running locally)

### Ollama Models Setup
This application requires the following models to be pulled into your local Ollama instance:
```bash
# Model for generating embeddings
ollama pull nomic-embed-text

# Model for chat/generation
ollama pull llama3.2
```

## Running the Application

### 1. Backend Setup

Open a terminal and navigate to the `backend` directory:
```bash
cd backend
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
pip install langchain-ollama langchain-chroma
```

Start the FastAPI development server:
```bash
uvicorn main:app --reload
```
The backend API will be available at `http://localhost:8000`.

### 2. Frontend Setup

Open a new terminal and navigate to the `frontend` directory:
```bash
cd frontend
```

Install the Node.js dependencies:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```
The frontend application will be available at `http://localhost:5173`.

## Features
- **PDF Upload**: Upload any PDF document to be parsed and ingested into the vector database.
- **Context-Aware Chat**: Ask questions, and the LLM will generate answers based on the context retrieved from your uploaded documents.
- **Persistent Storage**: Embeddings are stored locally in `backend/chroma_db` using ChromaDB for persistent access across restarts.
