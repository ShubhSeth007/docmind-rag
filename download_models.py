from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading embedding model...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading reranker model...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Models downloaded successfully.")
