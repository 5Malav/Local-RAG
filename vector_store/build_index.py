import pickle
from pathlib import Path
import faiss
from tqdm import tqdm

# ---------------------------
# CONFIGURATION
# ---------------------------
EMBEDDINGS_FILE = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag/embeddings/all_embeddings.pkl")  # embeddings
VECTOR_STORE_FOLDER = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag/vector_store")
VECTOR_STORE_FOLDER.mkdir(exist_ok=True)

FAISS_INDEX_FILE = VECTOR_STORE_FOLDER / "faiss_index.index"
METADATA_FILE = VECTOR_STORE_FOLDER / "metadata.pkl"

# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------
print("[INFO] Loading embeddings...")
with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings_data = pickle.load(f)

print(f"[INFO] Total chunks to index: {len(embeddings_data)}")

# ---------------------------
# EXTRACT VECTORS AND METADATA
# ---------------------------
vectors = []
metadata = []

for item in tqdm(embeddings_data):
    vectors.append(item["embedding"])  # the vector
    metadata.append({
        "doc_name": item["doc_name"],
        "chunk_id": item.get("chunk_id"),
        "text": item["text"]
    })

# Convert to float32 numpy array for FAISS
import numpy as np
vectors = np.array(vectors).astype("float32")
dimension = vectors.shape[1]
print(f"[INFO] Embedding dimension: {dimension}")

# ---------------------------
# BUILD FAISS INDEX
# ---------------------------
print("[INFO] Building FAISS index...")
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(vectors)
print(f"[INFO] Total vectors indexed: {index.ntotal}")

# ---------------------------
# SAVE INDEX AND METADATA
# ---------------------------
faiss.write_index(index, str(FAISS_INDEX_FILE))
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"[INFO] FAISS index saved to: {FAISS_INDEX_FILE}")
print(f"[INFO] Metadata saved to: {METADATA_FILE}")