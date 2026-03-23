import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# ---------------------------
# Project Paths
# ---------------------------
PROJECT_ROOT = Path("/Users/malavjoshi/Desktop/RAG_Projects/local_rag")
METADATA_PATH = PROJECT_ROOT / "vector_store/metadata.pkl"
INDEX_PATH = PROJECT_ROOT / "vector_store/faiss_index.index"

# ---------------------------
# Load Metadata
# ---------------------------
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# ---------------------------
# Initialize Embedding Model
# ---------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Evaluation Functions
# ---------------------------
def retrieval_accuracy(query, retrieved_chunks, top_k=5):
    """
    Evaluate retrieval: Compute cosine similarity of query vs retrieved chunks.
    """
    query_vec = embedding_model.encode([query])
    scores = []
    for chunk in retrieved_chunks:
        chunk_vec = embedding_model.encode([chunk["text"]])
        sim = cosine_similarity(query_vec, chunk_vec)[0][0]
        scores.append(sim)
    avg_score = sum(scores) / len(scores)
    print(f"[Retrieval] Average Cosine Similarity (Top-{top_k}): {avg_score:.4f}")
    return avg_score

def generation_quality(predicted_answer, reference_answer):
    """
    Evaluate generated answer using ROUGE and BLEU
    """
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_answer, predicted_answer)

    # BLEU
    reference_tokens = [reference_answer.split()]
    predicted_tokens = predicted_answer.split()
    bleu_score = sentence_bleu(reference_tokens, predicted_tokens)

    print("\n[Generation] Evaluation Metrics")
    print("ROUGE-1:", rouge_scores['rouge1'].fmeasure)
    print("ROUGE-L:", rouge_scores['rougeL'].fmeasure)
    print("BLEU:", bleu_score)

    return rouge_scores, bleu_score

# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    from retrieval.retriever import Retriever
    from generation.generator import Generator

    # Example query and expected answer
    query = "What is retrieval augmented generation?"
    expected_answer = ("Retrieval-Augmented Generation (RAG) is a technique where a language "
                       "model retrieves relevant documents from a knowledge base to generate "
                       "more accurate and factual responses.")

    # Load retriever & generator
    retriever = Retriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    generator = Generator(model_name="mistral:latest",
                          index_path=INDEX_PATH,
                          metadata_path=METADATA_PATH)

    # Step 1: Retrieval evaluation
    top_chunks = retriever.search(query, top_k=5)
    retrieval_accuracy(query, top_chunks, top_k=5)

    # Step 2: Generation evaluation
    generated_answer = generator.generate(query)
    generation_quality(generated_answer, expected_answer)