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
```

2. **Create a Python virtual environment**

```bash
python -m venv venv
# Activate on macOS/Linux
source venv/bin/activate
# Activate on Windows
venv\Scripts\activate
```
3. **install required packages:**

```bash
pip install -r requirements.txt
```

4. **Ensure your local LLM (Mistral 7B) is downloaded and accessible:**

```bash
local_rag/local_models/mistral-7b-instruct/
```
## Usage

1. **Preprocess documents**
 
```bash
python ingestion_pipeline/loader.py
python ingestion_pipeline/chunker.py
```

2. **Generate embeddings**

```bash
python embeddings/embed.py
```

3. **Build vector store**

```bash
python vector_store/build_index.py
```

4. **Query the system**
```bash
python generation/generator.py
```
  **CLI:**
```bash
python ui/cli.py --question "What is reinforcement learning?" --top_k 5
```
**GUI (optional):**
```bash
python ui/gradio_ui.py
```

5. **Evaluate**

```bash
python evaluation/evaluate.py
```

**Example Query**

```bash
query = "What is an Operating System?"
generated_answer = generator.generate(query)
print(generated_answer)
```

**Sample Output:**

```bash
An operating system (OS) is system software that acts as an interface between computer hardware and users. 
It manages hardware resources and provides services for application programs. 
Common examples include Linux, Windows, and macOS.
```

## Evaluation Metrics:-
- Retrieval Accuracy: Cosine similarity between query and retrieved chunks
- Generation Quality: ROUGE-1, ROUGE-L, BLEU scores

### Example Output:

- [Retrieval] Average Cosine Similarity (Top-5): 0.3793
- [Generation] Evaluation Metrics
- ROUGE-1: 0.7563
- ROUGE-L: 0.6891
- BLEU: 0.2801

## Performance Notes
- Retrieval is very fast (~milliseconds)
- Generation may take a few seconds depending on top_k and hardware
- Adjust top_k for speed vs accuracy tradeoff

## Troubleshooting
- Ollama connection errors: Ensure Ollama daemon is running
- UNEXPECTED warnings: Can be ignored unless exact architecture match is required
- Missing Python modules: Run pip install -r requirements.txt
