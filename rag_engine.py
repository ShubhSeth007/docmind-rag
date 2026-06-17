"""
rag_engine.py — Core RAG logic.
Handles: PDF parsing, chunking, embedding, ChromaDB storage, retrieval.
Note: Cross-encoder reranker removed to stay within 512MB free tier RAM limit.
Cosine similarity retrieval with sentence-transformers is used instead.
"""

import os
import re
import logging

import fitz                          # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, ~80MB
CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_store")
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
TOP_K_RETRIEVE   = 5

# ── Lazy singletons ────────────────────────────────────────────────────────────
_embed_model    = None
_chroma_client  = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection(doc_id: str) -> chromadb.Collection:
    safe_name       = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)[:60]
    collection_name = f"doc_{safe_name}"
    return get_chroma_client().get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def list_documents() -> list[dict]:
    try:
        collections = get_chroma_client().list_collections()
        return [
            {
                "doc_id":      col.name.replace("doc_", "", 1),
                "collection":  col.name,
                "chunk_count": col.count(),
            }
            for col in collections
        ]
    except Exception as e:
        log.error(f"Failed to list documents: {e}")
        return []


# ── PDF → chunks ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_num, "text": text})
    log.info(f"Extracted {len(pages)} pages from PDF")
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    chunks      = []
    chunk_index = 0

    for page_data in pages:
        text     = page_data["text"]
        page_num = page_data["page"]
        start    = 0

        while start < len(text):
            end        = start + CHUNK_SIZE
            chunk      = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append({
                    "chunk_id":    f"p{page_num}_c{chunk_index}",
                    "text":        chunk,
                    "page":        page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    log.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ── Ingest pipeline ────────────────────────────────────────────────────────────
def ingest_document(doc_id: str, pdf_bytes: bytes) -> dict:
    log.info(f"[ingest] Starting for doc_id={doc_id}")

    pages = extract_text_from_pdf(pdf_bytes)
    if not pages:
        raise ValueError("No extractable text found in PDF.")

    chunks = chunk_text(pages)
    if not chunks:
        raise ValueError("No valid chunks created from PDF.")

    embed_model = get_embed_model()
    texts       = [c["text"] for c in chunks]
    log.info(f"[ingest] Embedding {len(texts)} chunks...")
    embeddings  = embed_model.encode(
        texts, batch_size=16, show_progress_bar=False
    ).tolist()

    collection = get_collection(doc_id)
    if collection.count() > 0:
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
        "doc_id":    doc_id,
        "pages":     len(pages),
        "chunks":    len(chunks),
        "embed_dim": len(embeddings[0]),
    }


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_and_rerank(doc_id: str, question: str) -> list[dict]:
    """
    Embeds the question and retrieves the top-K most similar chunks
    via cosine similarity search in ChromaDB.
    """
    embed_model = get_embed_model()
    q_embedding = embed_model.encode(
        [question], show_progress_bar=False
    ).tolist()

    collection = get_collection(doc_id)
    if collection.count() == 0:
        raise ValueError(
            f"No chunks found for doc_id='{doc_id}'. "
            "Did you upload this document?"
        )

    n = min(TOP_K_RETRIEVE, collection.count())
    results = collection.query(
        query_embeddings = q_embedding,
        n_results        = n,
        include          = ["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":  doc,
            "page":  meta.get("page", 0),
            "score": round(float(1 - dist), 4),   # convert distance → similarity
        })

    return chunks
