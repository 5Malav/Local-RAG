# gradio_ui.py
import gradio as gr
from pathlib import Path
import sys

# Project root
PROJECT_ROOT = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag")
sys.path.append(str(PROJECT_ROOT))
INDEX_PATH = PROJECT_ROOT / "vector_store/faiss_index.index"
METADATA_PATH = PROJECT_ROOT / "vector_store/metadata.pkl"

from generation.generator import Generator
# Initialize generator
generator = Generator(
    model_name="mistral:latest",
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

# Function to wrap generator for Gradio
def answer_question(query, top_k=5):
    response = generator.generate(query, top_k=top_k)
    return response

# Gradio Interface
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Ask your question"),
        gr.Slider(1, 10, value=5, label="Top-K chunks")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Local Pdfs loaded for Question Answers",
    description="Ask questions to your local RAG model!"
)

if __name__ == "__main__":
    iface.launch()