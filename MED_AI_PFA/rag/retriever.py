import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(_PROJECT_ROOT, "Data" ,"Extracted_Data", "csv_faiss.index")
METADATA_PATH    = os.path.join(_PROJECT_ROOT, "Data" ,"Extracted_Data", "csv_metadata.json")
EMBEDDING_MODEL  = "BAAI/bge-small-en"
TOP_K            = 5

# ==============================
# LAZY-LOADED RESOURCES
# (loaded once on first call, safe to import without side-effects)
# ==============================

_model    = None
_index    = None
_metadata = None


def _load():
    """Load embedding model, FAISS index, and metadata (once)."""
    global _model, _index, _metadata
    if _model is not None:
        return  # already loaded

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retriever] Using device: {device}")

    print("[retriever] Loading embedding model...")
    _model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("[retriever] Loading FAISS index...")
    _index = faiss.read_index(FAISS_INDEX_PATH)

    print("[retriever] Loading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    print(f"[retriever] Ready — {_index.ntotal} vectors | {len(_metadata)} metadata entries")


# ==============================
# RETRIEVAL FUNCTION
# ==============================

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed *question* and return the *top_k* most similar chunks
    from the FAISS index, each as a metadata dict augmented with
    a 'score' (cosine similarity) field.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    top_k : int
        Number of chunks to return (default: 5).

    Returns
    -------
    list[dict]
        Ordered list of the top-k matching chunk metadata dicts,
        most similar first.
    """
    _load()   # no-op if already loaded

    query_embedding = _model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = _index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = _metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results


# ==============================
# PRETTY PRINT HELPER
# ==============================

def print_results(results: list[dict]) -> None:
    for rank, chunk in enumerate(results, start=1):
        print(f"{'='*60}")
        print(f"Rank #{rank}  |  Score: {chunk['score']:.4f}")
        print(f"Source : {chunk.get('source_name', 'N/A')}  |  Page: {chunk.get('page', 'N/A')}  |  Section: {chunk.get('section', 'N/A')}")
        content = chunk.get("content", "")
        print(f"Content:\n{content[:400]}{'...' if len(content) > 400 else ''}")
    print("="*60)


# ==============================
# MAIN — interactive demo
# ==============================

if __name__ == "__main__":
    print("MedQuad RAG Retriever — type 'quit' to exit\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        results = retrieve(question)
        print(f"\nTop {TOP_K} most relevant chunks:\n")
        print_results(results)
        print()