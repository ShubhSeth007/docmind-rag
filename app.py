"""
app.py — DocMind RAG API
Endpoints: /upload, /ask, /documents, /health, /metrics
"""

import os
import uuid
import time
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from rag_engine import ingest_document, retrieve_and_rerank, list_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Groq client (lazy) ─────────────────────────────────────────────────────────
_groq_client = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocMind — RAG Document Q&A API",
    description=(
        "Upload any PDF and ask questions about it. "
        "Powered by sentence-transformers embeddings, ChromaDB vector search, "
        "cross-encoder reranking, and Groq LLM (llama-3.3-70b)."
    ),
    version="1.0.0",
)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
UPLOADS_TOTAL = Counter(
    "docmind_uploads_total", "Total PDF uploads", ["status"]
)
QUESTIONS_TOTAL = Counter(
    "docmind_questions_total", "Total questions asked", ["status"]
)
UPLOAD_LATENCY = Histogram(
    "docmind_upload_latency_seconds", "PDF ingestion latency",
    buckets=[1, 2, 5, 10, 20, 30, 60],
)
QUESTION_LATENCY = Histogram(
    "docmind_question_latency_seconds", "Question answering latency",
    buckets=[0.5, 1, 2, 3, 5, 10, 20],
)
DOCS_STORED = Gauge(
    "docmind_documents_stored", "Number of documents in the vector store"
)


# ── Request / response schemas ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    doc_id:   str  = Field(..., description="Document ID returned by /upload")
    question: str  = Field(..., min_length=3, max_length=1000)
    top_k:    int  = Field(5, ge=1, le=10, description="Number of chunks to retrieve")


class AskResponse(BaseModel):
    answer:     str
    sources:    list[dict]   # [{"page": int, "excerpt": str, "score": float}]
    doc_id:     str
    latency_ms: float


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} — Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return f"""You are a precise document analyst. Answer the question using ONLY the provided context below.

Rules:
- Answer directly and concisely based on the context.
- If the context does not contain enough information, say: "The document does not contain sufficient information to answer this question."
- Always cite which page(s) your answer comes from at the end, like: (Source: Page 3, Page 7)
- Do not make up information not present in the context.

Context:
{context}

Question: {question}

Answer:"""


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    docs = list_documents()
    DOCS_STORED.set(len(docs))
    return {
        "status":         "operational",
        "documents_stored": len(docs),
        "groq_key_set":   bool(os.getenv("GROQ_API_KEY")),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the vector store.
    Returns a doc_id you use to ask questions.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 10MB limit
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    # Stable doc_id from filename so re-uploading same file overwrites cleanly
    safe_name = Path(file.filename).stem[:40]
    doc_id    = f"{safe_name}_{uuid.uuid4().hex[:8]}"

    start = time.perf_counter()
    try:
        summary = ingest_document(doc_id, contents)
        latency = time.perf_counter() - start
        UPLOADS_TOTAL.labels(status="success").inc()
        UPLOAD_LATENCY.observe(latency)
        DOCS_STORED.set(len(list_documents()))

        return {
            "doc_id":      doc_id,
            "filename":    file.filename,
            "pages":       summary["pages"],
            "chunks":      summary["chunks"],
            "embed_dim":   summary["embed_dim"],
            "latency_ms":  round(latency * 1000, 2),
            "message":     f"Document ingested. Use doc_id '{doc_id}' to ask questions.",
        }
    except ValueError as e:
        UPLOADS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        UPLOADS_TOTAL.labels(status="error").inc()
        log.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Ask a question about an uploaded document.
    Returns the answer with source page citations.
    """
    start = time.perf_counter()

    try:
        # 1. Retrieve + rerank
        chunks = retrieve_and_rerank(req.doc_id, req.question)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No content found for doc_id='{req.doc_id}'. Upload the document first."
            )

        # 2. Build prompt and call Groq
        prompt = build_prompt(req.question, chunks)
        groq   = get_groq()

        completion = groq.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.1,     # low temp = factual, less hallucination
            max_tokens  = 1024,
        )
        answer = completion.choices[0].message.content.strip()

        latency = time.perf_counter() - start
        QUESTIONS_TOTAL.labels(status="success").inc()
        QUESTION_LATENCY.observe(latency)

        # 3. Build source citations
        sources = [
            {
                "page":    chunk["page"],
                "excerpt": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "score":   chunk["score"],
            }
            for chunk in chunks
        ]

        return AskResponse(
            answer     = answer,
            sources    = sources,
            doc_id     = req.doc_id,
            latency_ms = round(latency * 1000, 2),
        )

    except HTTPException:
        raise
    except ValueError as e:
        QUESTIONS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        QUESTIONS_TOTAL.labels(status="error").inc()
        log.error(f"Question failed: {e}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")


@app.get("/documents")
def get_documents():
    """List all documents currently stored in the vector store."""
    docs = list_documents()
    DOCS_STORED.set(len(docs))
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all its chunks from the vector store."""
    try:
        global _chroma_client
        from rag_engine import get_collection, _chroma_client as cc
        import chromadb
        from chromadb.config import Settings
        from rag_engine import CHROMA_PATH

        if cc is None:
            raise HTTPException(status_code=404, detail="No documents stored yet.")

        collection_name = f"doc_{doc_id}"
        try:
            cc.delete_collection(collection_name)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

        DOCS_STORED.set(len(list_documents()))
        return {"message": f"Document '{doc_id}' deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
