import os
from pathlib import Path
from PyPDF2 import PdfReader

# ---------------------------
# CONFIGURATION
# ---------------------------
PDF_FOLDER = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag/data/pdfs")   # Folder containing PDFs
TXT_FOLDER = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag/data/txt")    # Folder containing TXT files
CHUNK_SIZE = 500                     # Approximate words per chunk

# ---------------------------
# FUNCTIONS
# ---------------------------

# Load PDF and extract text
def load_pdf(file_path):
    """Read a PDF file and return all text as a single string."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # only add if text exists
            text += page_text + "\n"
    return text

# Load TXT and extract text
def load_txt(file_path):
    """Read a TXT file and return all text as a single string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Split text into smaller chunks
def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Split a long text into smaller chunks of 'chunk_size' words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Main ingestion function
def ingest_documents():
    """
    Process all PDF and TXT files in the data folders.
    Returns a list of dictionaries containing:
    - doc_name: file name
    - chunk_id: automatic ID for each chunk
    - text: chunk text
    """
    all_chunks = []

    # Process PDFs
    for pdf_file in PDF_FOLDER.glob("*.pdf"):
        text = load_pdf(pdf_file)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_name": pdf_file.name,
                "chunk_id": idx,  # automatic chunk ID
                "text": chunk
            })

    # Process TXT files
    for txt_file in TXT_FOLDER.glob("*.txt"):
        text = load_txt(txt_file)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_name": txt_file.name,
                "chunk_id": idx,  # automatic chunk ID
                "text": chunk
            })

    print(f"[INFO] Total chunks created: {len(all_chunks)}")
    return all_chunks

# ---------------------------
# MAIN TEST
# ---------------------------
if __name__ == "__main__":
    chunks = ingest_documents()
    print("Sample chunk from first document:")
    print(f"Document: {chunks[3]['doc_name']}, Chunk ID: {chunks[1]['chunk_id']}")
    print(chunks[1]['text'][:500] + "...")