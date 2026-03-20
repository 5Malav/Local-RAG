# Vector Store (FAISS)

This module builds a FAISS index from document embeddings and enables similarity search.

## Files

- `build_index.py` → Builds FAISS index
- `faiss_index.index` → Stored vector index (ignored in Git)
- `metadata.pkl` → Chunk metadata (ignored in Git)
- `test_search.py` → Test retrieval

---

## Demo Query

# Vector Store (FAISS)

This module builds a FAISS index from document embeddings and enables fast similarity-based retrieval.

---

## Overview

- Embeddings are generated from document chunks
- FAISS is used to index and search vectors
- Returns top-k most relevant chunks for a given query

---

## Files

- `build_index.py` → Builds FAISS index
- `test_search.py` → Runs demo queries
- `faiss_index.index` → Stored index (ignored in Git)
- `metadata.pkl` → Chunk metadata (ignored in Git)

---

## Demo Queries & Results

### Query 1: What is deep learning?

**Top Results:**

1. **Document:** deep_learning.pdf  
   **Chunk ID:** 0  
   **Text:** Deep learning is a specialized branch of machine learning that uses neural networks with multiple layers to model complex patterns in data...

2. **Document:** machine_learning.pdf  
   **Chunk ID:** 0  
   **Text:** Machine learning is a branch of artificial intelligence that focuses on enabling computers to learn patterns from data...

3. **Document:** deep_learning.pdf  
   **Chunk ID:** 1  
   **Text:** Applications of deep learning include computer vision, machine translation, and speech recognition...

---

### Query 2: What is Operating System?

**Top Results:**

1. **Document:** operating_system.pdf  
   **Chunk ID:** 0  
   **Text:** An operating system (OS) is system software that acts as an interface between computer hardware and users...

2. **Document:** operating_system.pdf  
   **Chunk ID:** 1  
   **Text:** Types include time-sharing systems, distributed systems, and real-time systems...

3. **Document:** internet.txt  
   **Chunk ID:** 0  
   **Text:** Internet architecture refers to the design and structure of interconnected computer networks...

---

### Query 3: What is RAG and vector databases?

**Top Results:**

1. **Document:** vector_databases.txt  
   **Chunk ID:** 1  
   **Text:** Steps in RAG include generating embeddings, storing them in a vector database, and retrieving nearest neighbors...

2. **Document:** vector_databases.txt  
   **Chunk ID:** 0  
   **Text:** Vector databases are designed to store and query high-dimensional vector data...

3. **Document:** rag.txt  
   **Chunk ID:** 0  
   **Text:** Retrieval-Augmented Generation (RAG) combines information retrieval with generative AI...

---

## How to Run

```bash
python build_index.py
python test_search.py