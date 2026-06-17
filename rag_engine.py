"""
rag_engine.py — Core RAG logic.
Handles: PDF parsing, chunking, embedding, ChromaDB storage, retrieval, reranking.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import fitz                          # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, ~80MB
RERANK_MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # reranker
CHROMA_PATH        = os.getenv("CHROMA_PATH", "./chroma_store")
CHUNK_SIZE         = 512     # characters (not tokens — simpler, works well)
CHUNK_OVERLAP      = 64
TOP_K_RETRIEVE     = 10      # how many chunks to retrieve before reranking
TOP_K_RERANK       = 5       # how many chunks to pass to the LLM after reranking


# ── Lazy singletons ────────────────────────────────────────────────────────────
_embed_model  = None
_rerank_model = None
_chroma_client = None
_chroma_collection = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        log.info(f"Loading reranker: {RERANK_MODEL_NAME}")
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model


def get_collection(doc_id: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a specific document."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    # Each uploaded document gets its own collection (clean separation)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)[:60]
    collection_name = f"doc_{safe_name}"
    return _chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def list_documents() -> list[dict]:
    """List all ingested documents from ChromaDB."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    try:
        collections = _chroma_client.list_collections()
        docs = []
        for col in collections:
            count = col.count()
            docs.append({
                "doc_id": col.name.replace("doc_", "", 1),
                "collection": col.name,
                "chunk_count": count,
            })
        return docs
    except Exception as e:
        log.error(f"Failed to list documents: {e}")
        return []


# ── PDF → chunks ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text page by page from a PDF.
    Returns list of {"page": int, "text": str}.
    """
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_num, "text": text})
    log.info(f"Extracted {len(pages)} pages from PDF")
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Sliding window chunker — splits text into overlapping chunks.
    Each chunk carries its source page number for citation.
    Returns list of {"chunk_id": str, "text": str, "page": int, "chunk_index": int}.
    """
    chunks = []
    chunk_index = 0

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:   # skip tiny fragments
                chunks.append({
                    "chunk_id":    f"p{page_num}_c{chunk_index}",
                    "text":        chunk_text,
                    "page":        page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    log.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ── Ingest pipeline ────────────────────────────────────────────────────────────
def ingest_document(doc_id: str, pdf_bytes: bytes) -> dict:
    """
    Full ingestion pipeline:
    PDF bytes → pages → chunks → embeddings → ChromaDB
    Returns summary dict.
    """
    log.info(f"[ingest] Starting ingestion for doc_id={doc_id}")

    # 1. Extract text
    pages = extract_text_from_pdf(pdf_bytes)
    if not pages:
        raise ValueError("No extractable text found in PDF. Is it a scanned image-only PDF?")

    # 2. Chunk
    chunks = chunk_text(pages)
    if not chunks:
        raise ValueError("PDF text was extracted but no valid chunks were created.")

    # 3. Embed
    embed_model = get_embed_model()
    texts = [c["text"] for c in chunks]
    log.info(f"[ingest] Embedding {len(texts)} chunks...")
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    # 4. Store in ChromaDB
    collection = get_collection(doc_id)

    # Clear existing data for this doc (handles re-upload)
    existing = collection.count()
    if existing > 0:
        log.info(f"[ingest] Clearing {existing} existing chunks for doc_id={doc_id}")
        collection.delete(where={"doc_id": {"$eq": doc_id}})

    collection.add(
        ids        = [c["chunk_id"] for c in chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [
            {
                "doc_id":      doc_id,
                "page":        c["page"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )

    log.info(f"[ingest] Done. {len(chunks)} chunks stored.")
    return {
        "doc_id":      doc_id,
        "pages":       len(pages),
        "chunks":      len(chunks),
        "embed_dim":   len(embeddings[0]),
    }


# ── Retrieval + reranking pipeline ────────────────────────────────────────────
def retrieve_and_rerank(doc_id: str, question: str) -> list[dict]:
    """
    1. Embed the question
    2. Cosine similarity search → top-K chunks
    3. Cross-encoder rerank → top-5 chunks
    Returns list of {"text": str, "page": int, "score": float}
    """
    # 1. Embed question
    embed_model = get_embed_model()
    q_embedding = embed_model.encode([question], show_progress_bar=False).tolist()

    # 2. Vector search
    collection = get_collection(doc_id)
    if collection.count() == 0:
        raise ValueError(f"No chunks found for doc_id={doc_id}. Did you upload this document?")

    results = collection.query(
        query_embeddings = q_embedding,
        n_results        = min(TOP_K_RETRIEVE, collection.count()),
        include          = ["documents", "metadatas", "distances"],
    )

    raw_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        raw_chunks.append({
            "text":     doc,
            "page":     meta.get("page", 0),
            "distance": dist,
        })

    if not raw_chunks:
        return []

    # 3. Cross-encoder rerank
    reranker    = get_rerank_model()
    pairs       = [(question, c["text"]) for c in raw_chunks]
    rerank_scores = reranker.predict(pairs).tolist()

    for chunk, score in zip(raw_chunks, rerank_scores):
        chunk["score"] = round(float(score), 4)

    reranked = sorted(raw_chunks, key=lambda x: x["score"], reverse=True)
    return reranked[:TOP_K_RERANK]
