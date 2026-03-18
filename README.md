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