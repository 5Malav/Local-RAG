from pathlib import Path
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# PATH SETUP
# ---------------------------
PROJECT_ROOT = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag")
VECTOR_STORE = PROJECT_ROOT / "vector_store"

FAISS_INDEX_FILE = VECTOR_STORE / "faiss_index.index"
METADATA_FILE = VECTOR_STORE / "metadata.pkl"
# ---------------------------
# LOAD INDEX + METADATA
# ---------------------------
index = faiss.read_index(str(FAISS_INDEX_FILE))

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

# ---------------------------
# LOAD EMBEDDING MODEL
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# QUERY -- 1
# ---------------------------
query_1 = "What is the difference between deep learning and machine learning?"

query_vector_1 = model.encode([query_1]).astype("float32")

# ---------------------------
# SEARCH TOP-K
# ---------------------------
k = 3
distances, indices = index.search(query_vector_1, k)

print("\n[QUERY 1]:", query_1)
print("\nTop Results:\n")

for i, idx in enumerate(indices[0]):
    result = metadata[idx]
    print(f"Rank {i+1}")
    print(f"Document: {result['doc_name']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text'][:200]}...\n")

# ---------------------------
# QUERY -- 2
# ---------------------------
query_2 = "What is Operating System?"

query_vector_2 = model.encode([query_2]).astype("float32")

# ---------------------------
# SEARCH TOP-K
# ---------------------------
k = 3
distances, indices = index.search(query_vector_2, k)

print("\n[QUERY 2]:", query_2)
print("\nTop Results:\n")

for i, idx in enumerate(indices[0]):
    result = metadata[idx]
    print(f"Rank {i+1}")
    print(f"Document: {result['doc_name']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text'][:200]}...\n")

# ---------------------------
# QUERY -- 3    
# ---------------------------
query_3 = "What is rag and vector databases?"

query_vector_3 = model.encode([query_3]).astype("float32")

# ---------------------------
# SEARCH TOP-K
# ---------------------------
k = 3
distances, indices = index.search(query_vector_3, k)

print("\n[QUERY 3]:", query_3)
print("\nTop Results:\n")

for i, idx in enumerate(indices[0]):
    result = metadata[idx]
    print(f"Rank {i+1}")
    print(f"Document: {result['doc_name']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text'][:200]}...\n")