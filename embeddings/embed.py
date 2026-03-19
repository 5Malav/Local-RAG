import sys
from pathlib import Path
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add project root to Python path
PROJECT_ROOT = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag")
sys.path.append(str(PROJECT_ROOT))

# Import your ingestion function
from ingestion_pipeline.ingestion import ingest_documents
# ---------------------------
# CONFIGURATION
# ---------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model
EMBEDDINGS_FOLDER = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag/embeddings")
EMBEDDINGS_FOLDER.mkdir(exist_ok=True)  # create folder if not exists

# ---------------------------
# LOAD CHUNKS
# ---------------------------
print("[INFO] Loading documents and splitting into chunks...")
chunks = ingest_documents()  # returns list of dicts with doc_name, chunk_id, text

# ---------------------------
# INITIALIZE EMBEDDING MODEL
# ---------------------------
print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# ---------------------------
# CREATE EMBEDDINGS
# ---------------------------
embeddings_data = []  # will store embeddings + metadata

print("[INFO] Generating embeddings...")
for chunk in tqdm(chunks):
    vector = model.encode(chunk["text"])
    embeddings_data.append({
        "doc_name": chunk["doc_name"],
        "chunk_id": chunk.get("chunk_id"),  # optional if you are using it
        "text": chunk["text"],
        "embedding": vector
    })

# ---------------------------
# SAVE EMBEDDINGS
# ---------------------------
output_file = EMBEDDINGS_FOLDER / "all_embeddings.pkl"
with open(output_file, "wb") as f:
    pickle.dump(embeddings_data, f)

print(f"[INFO] Embeddings saved to {output_file}")
print(f"[INFO] Total embeddings created: {len(embeddings_data)}")