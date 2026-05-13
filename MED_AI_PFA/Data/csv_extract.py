"""
Improved CSV-only Medical Vector Database Builder
==================================================
Processes MedQuAD CSV, cleans and enriches each QA pair,
embeds with a configurable sentence transformer,
and stores vectors in a FAISS index (cosine similarity).

Usage:
    python csv_extract.py
"""

import os
import sys
import json
import re
import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIGURATION – ADJUST PATHS
# ==============================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_BASE_DIR, "Data", "medquad.csv")
OUTPUT_FOLDER = os.path.join(_BASE_DIR, "Data", "Extracted_Data")
BATCH_SIZE = 32                     # embeddings per batch (increase if GPU memory allows)
EMBEDDING_MODEL = "BAAI/bge-small-en"   # good balance of speed/quality; use "BAAI/bge-large-en" for better results
CLEAN_TEXT = True                   # remove extra whitespace and normalise newlines
INCLUDE_FOCUS_IN_TEXT = True        # prepend focus_area to the document text

# ==============================
# ENVIRONMENT SETUP
# ==============================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
torch.manual_seed(42)               # for reproducibility

# ==============================
# DEVICE DETECTION
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==============================
# TEXT CLEANING FUNCTION
# ==============================

def clean_text(text: str) -> str:
    """Normalise whitespace, remove excessive newlines, and strip."""
    if not isinstance(text, str):
        return ""
    # Replace multiple newlines/spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any remaining control characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    return text.strip()

# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("\nLoading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model dimension: {embedding_dim}")

# ==============================
# FAISS INDEX (Cosine similarity via inner product on normalized vectors)
# ==============================

index_cpu = faiss.IndexFlatIP(embedding_dim)
index = index_cpu
use_gpu_for_faiss = False
if device == "cuda":
    try:
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            use_gpu_for_faiss = True
            print("FAISS index running on GPU.")
        else:
            print("FAISS GPU support not found – using CPU index.")
    except Exception as e:
        print(f"Could not initialize FAISS GPU: {e}\nFalling back to CPU.")
else:
    print("FAISS index running on CPU.")

# ==============================
# PROCESS CSV
# ==============================

print("\n--- Processing MedQuAD CSV ---")
if not os.path.exists(CSV_PATH):
    print(f"CSV not found: {CSV_PATH} – exiting.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
source_file = os.path.basename(CSV_PATH)
total_rows = len(df)
print(f"Found {total_rows} rows.")

documents = []
metadata = []

print("Preparing rows...")
for i, row in tqdm(df.iterrows(), total=total_rows, desc="Preparing"):
    # Extract and clean fields
    question = clean_text(str(row.get("question", "")))
    answer = clean_text(str(row.get("answer", "")))
    focus = clean_text(str(row.get("focus_area", "")))
    source = clean_text(str(row.get("source", "")))

    # Skip rows where both question and answer are empty
    if not question and not answer:
        continue

    # Build document text
    if INCLUDE_FOCUS_IN_TEXT and focus:
        doc_parts = [f"Focus: {focus}", f"Question: {question}", f"Answer: {answer}"]
    else:
        doc_parts = [f"Question: {question}", f"Answer: {answer}"]

    if source:
        doc_parts.append(f"Source: {source}")

    doc_text = "\n".join(doc_parts)
    documents.append(doc_text)

    # Store rich metadata
    metadata.append({
        "vector_id": None,               # to be filled after embedding
        "source_type": "medquad",
        "source_name": source_file,
        "focus_area": focus,
        "question": question,
        "answer": answer,
        "source": source,
        "content": doc_text
    })

print(f"Prepared {len(documents)} valid QA pairs (filtered out empty rows).")

# ==============================
# EMBED IN BATCHES
# ==============================

print("Embedding rows...")
embeddings_list = []
failed_indices = []

for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Encoding batches"):
    batch_texts = documents[i:i+BATCH_SIZE]
    try:
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")
        embeddings_list.append(batch_embeddings)
    except Exception as e:
        print(f"Error encoding batch starting at index {i}: {e}")
        # Mark these indices as failed (will be skipped)
        failed_indices.extend(range(i, min(i+BATCH_SIZE, len(documents))))

if not embeddings_list:
    print("No embeddings created – exiting.")
    sys.exit(1)

# Combine all successful embeddings
embeddings = np.vstack(embeddings_list)

# Remove metadata entries corresponding to failed embeddings
# (failed_indices are global indices; we need to adjust after filtering)
# Simpler: rebuild metadata list from successful embeddings
successful_metadata = []
vector_id = 0
for idx, meta in enumerate(metadata):
    if idx not in failed_indices:
        meta["vector_id"] = vector_id
        successful_metadata.append(meta)
        vector_id += 1

metadata = successful_metadata

# Add to FAISS index
index.add(embeddings)
print(f"Added {vector_id} vectors to index.")

# ==============================
# SAVE RESULTS
# ==============================

print("\nSaving FAISS index...")
if use_gpu_for_faiss:
    index = faiss.index_gpu_to_cpu(index)
faiss.write_index(index, os.path.join(OUTPUT_FOLDER, "csv_faiss.index"))

print("Saving metadata...")
with open(os.path.join(OUTPUT_FOLDER, "csv_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Total vectors: {vector_id}")
print(f"Output saved to: {OUTPUT_FOLDER}")