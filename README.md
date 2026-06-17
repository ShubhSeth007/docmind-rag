# DocMind — RAG Document Q&A API

A production-grade Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF and ask questions about it — with exact page citations in every answer.

Built with **sentence-transformers** embeddings, **ChromaDB** vector search, and **Groq LLM** (llama-3.3-70b). Containerized with Docker and deployed live on Render.

🔴 **Live:** [https://docmind-rag-pmsv.onrender.com](https://docmind-rag-pmsv.onrender.com)  
📖 **Swagger UI:** [https://docmind-rag-pmsv.onrender.com/docs](https://docmind-rag-pmsv.onrender.com/docs)

> ⚠️ Hosted on Render free tier — first request after inactivity may take 30–50 seconds to cold start. Subsequent requests are fast.

---

## Architecture

```
PDF upload (POST /upload)
        │
        ▼
Text extraction — PyMuPDF
        │
        ▼
Chunking — 500 chars, 50 overlap sliding window
        │
        ▼
Embedding — all-MiniLM-L6-v2 (384-dim vectors)
        │
        ▼
ChromaDB vector store (persisted to disk)
        │
        ├─── Question arrives (POST /ask)
        │           │
        │           ▼
        │     Embed question (same model)
        │           │
        │           ▼
        │     Cosine similarity search → top-5 chunks
        │           │
        │           ▼
        │     Prompt builder (context + question)
        │           │
        │           ▼
        │     Groq LLM — llama-3.3-70b-versatile
        │           │
        │           ▼
        │     Answer + page citations returned
        │
        └─── /metrics → Prometheus scrape endpoint
```

---

## Stack

| Component | Tool |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| Chunking | Custom sliding window (500 chars, 50 overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB 0.4.24 (local, persisted) |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Serving | FastAPI + Uvicorn |
| Metrics | Prometheus client |
| Deploy | Docker → Render |

---

## Live API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| [`/health`](https://docmind-rag-pmsv.onrender.com/health) | GET | Service readiness + Groq key check |
| [`/docs`](https://docmind-rag-pmsv.onrender.com/docs) | GET | Interactive Swagger UI |
| `/upload` | POST | Upload a PDF for ingestion |
| `/ask` | POST | Ask a question, get answer + citations |
| `/documents` | GET | List all ingested documents |
| `/documents/{doc_id}` | DELETE | Remove a document |
| [`/metrics`](https://docmind-rag-pmsv.onrender.com/metrics) | GET | Prometheus scrape endpoint |

---

## API Usage

### 1. Upload a PDF

```bash
curl -X POST https://docmind-rag-pmsv.onrender.com/upload \
  -F "file=@your_document.pdf"
```

```json
{
  "doc_id": "your_document_a3f9c1b2",
  "filename": "your_document.pdf",
  "pages": 12,
  "chunks": 94,
  "embed_dim": 384,
  "latency_ms": 4821.3,
  "message": "Document ingested. Use doc_id 'your_document_a3f9c1b2' to ask questions."
}
```

### 2. Ask a question

```bash
curl -X POST https://docmind-rag-pmsv.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "your_document_a3f9c1b2",
    "question": "What are the main findings?",
    "top_k": 5
  }'
```

```json
{
  "answer": "The document highlights three main findings... (Source: Page 4, Page 7)",
  "sources": [
    {
      "page": 4,
      "excerpt": "Our analysis reveals that...",
      "score": 0.8432
    },
    {
      "page": 7,
      "excerpt": "The second key finding...",
      "score": 0.7891
    }
  ],
  "doc_id": "your_document_a3f9c1b2",
  "latency_ms": 4229.25
}
```

### 3. List stored documents

```bash
curl https://docmind-rag-pmsv.onrender.com/documents
```

---

## Real Output Example

Tested on a machine learning engineer resume:

**Question:** *"What is this document about?"*

**Answer:**
> This document appears to be a resume of a machine learning engineer and computer science undergraduate, highlighting their education, skills, experience, and achievements in the field of machine learning, data analytics, and web development. (Source: Page 1)

---

## Local Setup

### Prerequisites
- Python 3.10+
- Free [Groq API key](https://console.groq.com) — takes 2 minutes to get

### Run with Docker

```bash
git clone https://github.com/YOUR_USERNAME/docmind-rag.git
cd docmind-rag

docker build -t docmind:v1 .

docker run -p 8000:10000 \
  -e PORT=10000 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  docmind:v1
```

Visit `http://localhost:8000/docs`

### Run without Docker

```bash
pip install -r requirements.txt

export GROQ_API_KEY=your_groq_api_key_here
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |
| `PORT` | No | Server port (default: 10000) |
| `CHROMA_PATH` | No | ChromaDB storage path (default: ./chroma_store) |

---

## Project Structure

```
├── app.py                  # FastAPI — /upload /ask /documents /health /metrics
├── rag_engine.py           # PDF parsing, chunking, embedding, ChromaDB retrieval
├── download_models.py      # Pre-downloads ML models at Docker build time
├── requirements.txt        # Pinned dependencies
├── Dockerfile              # Single-worker container, models baked in at build
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml          # Lint check on every push
```

---

## Related Project

This project complements the **[Fraud Detection MLOps System](https://github.com/YOUR_USERNAME/fraud-monitoring-dashboard)** — together they demonstrate the full ML Engineer stack:

| Project | Domain | Techniques |
|---|---|---|
| Fraud Detection MLOps | Tabular ML | XGBoost, Optuna, Evidently, Prometheus, CI/CD retraining |
| DocMind RAG | NLP / LLMs | Embeddings, ChromaDB, vector search, Groq LLM, RAG |

**Live deployments:**
- Fraud API: [fraud-detection-api-wxa6.onrender.com](https://fraud-detection-api-wxa6.onrender.com)
- MLOps Dashboard: [fraud-metrics-dashboard.onrender.com](https://fraud-metrics-dashboard.onrender.com)
- DocMind RAG: [docmind-rag-pmsv.onrender.com](https://docmind-rag-pmsv.onrender.com)
