import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    Retrieval module for RAG pipeline.
    Converts query to embeddings, searches FAISS, and returns top-k chunks.
    """

    def __init__(self, index_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 3):
        """
        Search the FAISS index for the query and return top-k chunks with metadata.
        """
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            result = self.metadata[idx]
            results.append({
                "doc_name": result["doc_name"],
                "chunk_id": result["chunk_id"],
                "text": result["text"]
            })
        return results


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    retriever = Retriever(
        index_path="/Users/malavjoshi/Desktop/RAG_Projects/local_rag/vector_store/faiss_index.index",
        metadata_path="/Users/malavjoshi/Desktop/RAG_Projects/local_rag/vector_store/metadata.pkl"
    )

    query = "What is deep learning?"
    results = retriever.search(query, top_k=3)

    print(f"\n[QUERY]: {query}\n")
    for i, r in enumerate(results):
        print(f"Rank {i+1}")
        print(f"Document: {r['doc_name']}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Text: {r['text'][:300]}...\n")