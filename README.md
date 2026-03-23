# Local RAG Project

A local **Retrieval-Augmented Generation (RAG)** tool to answer questions and summarize documents.

## Features
- Load PDF, TXT, DOCX documents
- Chunk text with metadata
- Create embeddings using Mistral or Sentence Transformers
- Vector search with FAISS/Chroma
- Generate answers and summaries with a local LLM
- CLI or simple GUI interface

## Folder Structure
- `data/` - raw documents
- `ingestion_pipeline/` - loaders and chunkers
- `embeddings/` - stored embeddings
- `vector_store/` - vector index
- `retrieval/` - retrieval module
- `generation/` - LLM generation
- `evaluation/` - testing and evaluation scripts
- `ui/` - CLI or GUI scripts
- `scripts/` - helper scripts
- `notebooks/` - experimentation

## Tech Stack
- Python 3.10+
- SentenceTransformers for embeddings (all-MiniLM-L6-v2)
- FAISS for vector similarity search
- Mistral 7B Local LLM for answer generation
- Ollama (optional SDK for local LLM inference)
- ROUGE / BLEU for evaluation metrics
- Gradio (optional) for GUI interface
- Pickle for saving embeddings and metadata
  
## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/<your-username>/local_rag.git
cd local_rag

2. **Create a Python virtual environment:**

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. **Install required packages:**
pip install -r requirements.txt

4. **Ensure your local LLM is downloaded:**

Make sure mistral-7b-instruct is inside:

local_rag/local_models/mistral-7b-instruct/


