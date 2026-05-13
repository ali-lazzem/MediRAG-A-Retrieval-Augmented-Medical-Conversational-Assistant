import ollama
from typing import List, Dict

OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a helpful medical assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.
Be concise, accurate, and cite the source document when possible."""

def build_prompt(context_chunks: List[Dict], question: str, history: List[Dict] = None) -> str:
    """
    Format retrieved chunks, conversation history, and the question into a single user message.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source_name", "Unknown")
        src = chunk.get("source", "?")               # new source field
        text = chunk.get("content", "")
        context_parts.append(f"[Chunk {i} | Source: {source}, Source Type: {src}]\n{text}")
    context_str = "\n\n".join(context_parts)

    history_str = ""
    if history:
        history_lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_str = "\n".join(history_lines) + "\n\n"

    return (
        f"Context:\n{context_str}\n\n"
        f"{history_str}"
        f"Question: {question}\n\n"
        f"Answer:"
    )

def answer(context_chunks: List[Dict], question: str, history: List[Dict] = None) -> str:
    user_message = build_prompt(context_chunks, question, history)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()