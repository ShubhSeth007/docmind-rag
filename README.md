# DocMind — RAG Document Q&A API

A production-grade Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF and ask questions about it — with exact page citations in every answer.

Built with **sentence-transformers** embeddings, **ChromaDB** vector search, **cross-encoder reranking**, and **Groq LLM** (llama-3.3-70b). Containerized with Docker and deployed live on Render.

🔴 **Live:** [https://YOUR-SERVICE.onrender.com](https://YOUR-SERVICE.onrender.com)

---

## Architecture

```
PDF upload
    │
    ▼
Text extraction (PyMuPDF)
    │
    ▼
Chunking (512 chars, 64 overlap)
    │
    ▼
Embedding (all-MiniLM-L6-v2, 384-dim)
    │
    ▼
ChromaDB vector store (persisted)
    │
    ├─── Question comes in (POST /ask)
    │         │
    │         ▼
    │    Embed question (same model)
    │         │
    │         ▼
    │    Cosine similarity search → top-10 chunks
    │         │
    │         ▼
    │    Cross-encoder reranker → top-5 chunks
    │         │
    │         ▼
    │    Prompt builder (context + question)
    │         │
    │         ▼
    │    Groq LLM (llama-3.3-70b-versatile)
    │         │
    │         ▼
    │    Answer + page citations returned
    │
    └─── /metrics → Prometheus scrape endpoint
```

---

## What makes this different from basic RAG tutorials

**Cross-encoder reranking** — most RAG tutorials retrieve chunks by embedding similarity and stop there. This pipeline adds a cross-encoder reranker that rescores the top-10 retrieved chunks using full attention between the query and each chunk, yielding substantially more relevant context before the LLM call.

**Page citations** — every answer includes the exact page numbers it was drawn from, making the system auditable and trustworthy — a production requirement that most demos skip.

**Per-document collections** — each uploaded document gets its own ChromaDB collection, enabling multi-document support without cross-contamination.

---

## Stack

| Component | Tool |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| Chunking | Custom sliding window (512 chars, 64 overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB (local, persisted) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Serving | FastAPI + Uvicorn |
| Metrics | Prometheus client |
| Deploy | Docker → Render |

---

## API Endpoints

**Interactive Swagger UI:** [https://YOUR-SERVICE.onrender.com/docs](https://YOUR-SERVICE.onrender.com/docs)

### `GET /health`
```json
{
  "status": "operational",
  "documents_stored": 2,
  "groq_key_set": true
}
```

### `POST /upload`
Upload a PDF (max 10MB). Returns a `doc_id` to use for questions.

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/upload \
  -F "file=@your_document.pdf"
```

```json
{
  "doc_id": "my_report_a3f9c1b2",
  "filename": "my_report.pdf",
  "pages": 24,
  "chunks": 187,
  "embed_dim": 384,
  "latency_ms": 4821.3,
  "message": "Document ingested. Use doc_id 'my_report_a3f9c1b2' to ask questions."
}
```

### `POST /ask`
Ask a question about an uploaded document.

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "my_report_a3f9c1b2",
    "question": "What are the main findings of this report?",
    "top_k": 5
  }'
```

```json
{
  "answer": "The report identifies three main findings: ... (Source: Page 4, Page 7)",
  "sources": [
    {
      "page": 4,
      "excerpt": "Our analysis reveals that...",
      "score": 8.32
    },
    {
      "page": 7,
      "excerpt": "The second key finding...",
      "score": 6.91
    }
  ],
  "doc_id": "my_report_a3f9c1b2",
  "latency_ms": 1823.4
}
```

### `GET /documents`
List all documents in the vector store.

### `DELETE /documents/{doc_id}`
Remove a document and all its chunks.

### `GET /metrics`
Prometheus scrape endpoint exposing upload/question counters and latency histograms.

---

## Local Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (takes 30 seconds to get)

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

## Deployment on Render

1. Push repo to GitHub
2. Render → New → Web Service → connect repo
3. Runtime: **Docker**
4. Add environment variable: `GROQ_API_KEY = your_key`
5. Deploy

> Note: First build takes 6–8 minutes — the Dockerfile pre-downloads both ML models (~300MB) at build time so cold starts are instant.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |
| `PORT` | No | Server port (default: 10000) |
| `CHROMA_PATH` | No | ChromaDB storage path (default: ./chroma_store) |

---

## Related Project

This project complements the **[Fraud Detection MLOps System](https://github.com/YOUR_USERNAME/fraud-monitoring-dashboard)** — together they demonstrate the full ML Engineer stack:

| Project | Domain | Techniques |
|---|---|---|
| Fraud Detection | Tabular ML | XGBoost, Optuna, Evidently, Prometheus, CI/CD retraining |
| DocMind RAG | NLP / LLMs | Embeddings, vector search, reranking, RAG, Groq |
