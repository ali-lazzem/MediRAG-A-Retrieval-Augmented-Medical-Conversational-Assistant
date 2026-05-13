"""Global configuration for the comparison framework."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Paths to your existing FAISS index and metadata (from original project)
# Adjust these to point to your actual files
EXISTING_FAISS_INDEX = r"D:\# PROJECTS #\### PFA ###\evaluation vs other models\data\csv_faiss.index"
EXISTING_FAISS_METADATA = r"D:\# PROJECTS #\### PFA ###\evaluation vs other models\data\csv_metadata.json"

# Embedding models
DENSE_EMBEDDING_MODEL = "BAAI/bge-small-en"   # must match the existing index

# Evaluation embedding model (separate)
EVAL_EMBEDDING_MODEL = "BAAI/bge-small-en"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.0

# Retrieval
DEFAULT_TOP_K = 5
BATCH_SIZE = 32

# CSV columns (used for BM25 and fallback)
COL_QUESTION = "question"
COL_ANSWER = "answer"
COL_FOCUS = "focus_area"