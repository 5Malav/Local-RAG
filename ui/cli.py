# cli.py
import argparse
from pathlib import Path
import sys

# Project root
PROJECT_ROOT = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag")
sys.path.append(str(PROJECT_ROOT))

from generation.generator import Generator

def main(question=None, top_k=5):
    """
    Main function to run the CLI.
    If 'question' is provided, it uses that.
    Otherwise, it reads from command-line arguments.
    """
    if question is None:
        # Parse command-line arguments only if question is not provided
        parser = argparse.ArgumentParser(description="Local RAG CLI")
        parser.add_argument("--question", type=str, required=True, help="Question to ask the RAG system")
        parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
        args = parser.parse_args()
        question = args.question
        top_k = args.top_k

    # Paths for Retriever
    INDEX_PATH = PROJECT_ROOT / "vector_store/faiss_index.index"
    METADATA_PATH = PROJECT_ROOT / "vector_store/metadata.pkl"

    # Initialize Generator
    generator = Generator(
        model_name="mistral:latest",
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    # Generate answer
    answer = generator.generate(question, top_k=top_k)
    print("\n=== Answer ===")
    print(answer.replace(".", ".\n"))  # newline after each sentence

if __name__ == "__main__":
    main()
    
