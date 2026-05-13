Below is a complete README.md file for your project. It covers setup, configuration, usage, evaluation, and more.

markdown
# MediRAG – Medical Retrieval-Augmented Generation System

MediRAG is a production‑ready medical Q&A system that combines dense (FAISS) and sparse (BM25) retrieval with an LLM (Llama 3.2 via Ollama). It features a Django web interface, user authentication, conversation history, and comprehensive evaluation tools.

## Features

- **Dual retrieval engines** – FAISS (dense, BAAI/bge‑small‑en) and BM25 (sparse) for comparison.
- **Local LLM integration** – Uses Ollama to run Llama 3.2 locally – no external API calls.
- **Django web app** – User login, persistent chat sessions, dark mode, responsive design.
- **Evaluation suite** – Benchmark RAG performance with semantic similarity, ROUGE, BLEU, and BERTScore.
- **Pre‑built index** – FAISS index is built once from the MedQuAD CSV and reused.

## Tech Stack

| Component            | Technology                           |
|----------------------|--------------------------------------|
| Backend              | Django + Django REST Framework       |
| Vector search        | FAISS (CPU/GPU) + Sentence‑Transformers |
| Sparse search        | BM25 (rank‑bm25)                     |
| LLM                  | Ollama (Llama 3.2)                   |
| Embedding models     | BAAI/bge‑small‑en (retrieval & eval) |
| Frontend             | HTML/CSS/JS, FontAwesome, Google Fonts |
| Evaluation metrics   | Cosine similarity, ROUGE, BLEU, BERTScore |

## Project Structure
.
├── backend/ # Django project settings

│ ├── settings.py

│ ├── urls.py

│ └── ...

├── rag/ # Main Django app

│ ├── models.py # Conversation, Message, Profile

│ ├── views.py # Chat API, authentication views

│ ├── retriever.py # FAISS + BM25 retrieval

│ ├── llm.py # Ollama prompt builder

│ ├── retriever.py # Lazy‑loaded FAISS index

│ ├── urls.py

│ └── ...

├── templates/ # HTML templates

├── static/ # CSS, JS

├── Data/ # (create this)

│ ├── medquad.csv # MedQuAD dataset

│ └── Extracted_Data/ # FAISS index & metadata

├── csv_extract.py # Build FAISS index from CSV

├── benchmark.py # Compare RAG_FAISS vs RAG_BM25

├── evaluate.py # Detailed evaluation (multiple metrics)

├── requirements.txt

└── manage.py


text

## Setup Instructions

### 1. Clone the repository & create virtual environment

```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
2. Install dependencies
bash
pip install -r requirements.txt
3. Install Ollama and pull Llama 3.2
Download Ollama from ollama.ai and install.

Pull the model:

bash
ollama pull llama3.2
4. Prepare the MedQuAD dataset
Place the medquad.csv file inside the Data/ folder.
The CSV must contain at least these columns: question, answer, focus_area (optional), source (optional).

5. Build the FAISS index (one‑time)
bash
python csv_extract.py
This creates:

Data/Extracted_Data/csv_faiss.index

Data/Extracted_Data/csv_metadata.json

6. Run database migrations & create superuser
bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
7. Start the Django server
bash
python manage.py runserver
Open http://127.0.0.1:8000/ in your browser.

Configuration
All important settings are in config.py and the top of csv_extract.py.

Variable	Default	Description
EXISTING_FAISS_INDEX	Data/Extracted_Data/csv_faiss.index	Path to FAISS index
DENSE_EMBEDDING_MODEL	BAAI/bge-small-en	Model for retrieval
EVAL_EMBEDDING_MODEL	BAAI/bge-small-en	Model for evaluation
OLLAMA_MODEL	llama3.2	LLM used for generation
DEFAULT_TOP_K	5	Number of retrieved chunks
BATCH_SIZE	32	Batch size for embedding
Usage
Web Interface
Register / Login

Start a new consultation or continue previous ones

Ask medical questions – the system retrieves relevant Q&A pairs and generates an answer.

Dark mode toggle in the header.

API Endpoints
All API endpoints require authentication (session cookie).

Endpoint	Method	Description
/api/ask/	POST	Send a question, get answer + retrieved chunks. Body: {"question": "...", "top_k": 5, "session_id": "..."}
/api/sessions/	GET	List all conversations for the user
/api/sessions/create/	POST	Create a new session
/api/sessions/<id>/	GET	Get messages of a session
/api/sessions/<id>/rename/	PUT	Rename a session
/api/sessions/<id>/delete/	DELETE	Delete a session
Running Benchmarks
bash
# Compare FAISS vs BM25 on the curated test set (22 questions)
python benchmark.py --csv Data/medquad.csv --top_k 5

# Detailed evaluation with multiple metrics (ROUGE, BLEU, BERTScore)
python evaluate.py --top_k 5 --metrics all
Benchmark results are saved as benchmark_results_rag_only.json and evaluation outputs go to Data/evaluation/.

Evaluation Results (from benchmark_results_curated.json)
On a curated set of 22 medical questions:

Model	Mean Semantic Similarity	Avg Latency (s)
RAG_FAISS (bge‑small‑en)	0.918	23.5
RAG_BM25	0.891	50.2
FAISS provides both higher answer quality and faster retrieval.

Troubleshooting
FAISS index not found – Run csv_extract.py first and verify the path in config.py.

Ollama connection refused – Ensure Ollama is running (ollama serve).

GPU out of memory – Reduce BATCH_SIZE in csv_extract.py or use CPU.

Import errors – Make sure you activated the virtual environment and installed all requirements.

Customisation
Change embedding model – Edit DENSE_EMBEDDING_MODEL and EVAL_EMBEDDING_MODEL in config.py (must match the model used to build the FAISS index).

Switch LLM – Change OLLAMA_MODEL (e.g., mistral, llama3).

Add more test questions – Edit get_curated_questions() in data_loader.py.

Credits
MedQuAD dataset – National Library of Medicine (NIH)

FAISS – Meta Research

Sentence‑Transformers – UKPLab

Ollama – Ollama team

BAAI embedding models – Beijing Academy of Artificial Intelligence

License
MIT – feel free to use and modify.

text

Save this as `README.md` in your project root. Adjust paths or details as needed (e.g., if your CSV location differs). The file is ready to be copied and pasted.
