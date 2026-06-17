from sentence_transformers import SentenceTransformer

print("Downloading embedding model...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Models downloaded successfully.")
