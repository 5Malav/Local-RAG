# generator.py
# ---------------------------
# Imports
# ---------------------------
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag")
sys.path.append(str(PROJECT_ROOT))

# Your retriever module
from retrieval.retriever import Retriever

# Ollama SDK
import ollama

# ---------------------------
# Generator Class
# ---------------------------
class Generator:
    def __init__(self, model_name="mistral:latest", index_path=None, metadata_path=None):
        """
        Initialize the Generator with Ollama model and Retriever.
        """
        if index_path is None or metadata_path is None:
            raise ValueError("You must provide index_path and metadata_path for the Retriever.")

        # Load Retriever
        self.retriever = Retriever(index_path=index_path, metadata_path=metadata_path)
        print(f"[INFO] Retriever loaded with index: {index_path}")

        # Ollama model
        self.model_name = model_name
        print(f"[INFO] Generator initialized with Ollama model: {self.model_name}")

    def generate(self, query, top_k=5):
        """
        Generate an answer using retrieved chunks + Ollama LLM
        """
        # Step 1: Retrieve top-k relevant chunks
        docs = self.retriever.search(query, top_k=top_k)
        context = "\n".join([doc["text"] for doc in docs])

        # Step 2: Construct prompt
        prompt = f"Answer the following query based on context:\n\nContext:\n{context}\n\nQuery: {query}\nAnswer:"

        # Step 3: Generate response using Ollama LLM
        response = ollama.generate(model=self.model_name, prompt=prompt)

        # Step 4: Return the actual generated text
        return response.response
        
        

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Update these paths according to your project
    INDEX_PATH = PROJECT_ROOT / "vector_store/faiss_index.index"
    METADATA_PATH = PROJECT_ROOT / "vector_store/metadata.pkl"

    # Initialize generator
    generator = Generator(
        model_name="mistral:latest",
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    # Example query
    query = "What is internet?"
    answer = generator.generate(query)

    print("\n=== Generated Answer ===")
    print(answer.replace(".",".\n"))