FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so startup is instant on Render
# This bakes the models into the Docker image (~300MB) instead of
# downloading them on every cold start (which would time out)
RUN python -c " \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models downloaded successfully') \
"

# Copy application code
COPY app.py .
COPY rag_engine.py .

# ChromaDB persistence directory
RUN mkdir -p /app/chroma_store

ENV CHROMA_PATH=/app/chroma_store

EXPOSE 10000

# Single worker — keeps memory under 512MB free tier limit
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
