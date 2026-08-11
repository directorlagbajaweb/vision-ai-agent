"""
memory/semantic.py
Local, free semantic memory search using sentence-transformers embeddings
and a local chromadb vector store. Fully offline, no external API calls —
your conversation history never leaves your machine.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
import config

_client = chromadb.PersistentClient(path=str(config.BASE_DIR / "memory" / "chroma_store"))
_collection = _client.get_or_create_collection(name="conversation_memory")

print("[semantic] Loading local embedding model (first run may take a moment)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[semantic] Embedding model ready.")

_id_counter = 0


def add_message(role: str, content: str):
    """Embeds and stores a message for future semantic search."""
    global _id_counter
    if not content or not content.strip():
        return

    embedding = _model.encode(content).tolist()
    doc_id = f"{role}-{time.time()}-{_id_counter}"
    _id_counter += 1

    _collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"role": role}],
    )


def search_relevant(query: str, n_results: int = 5) -> list[dict]:
    """Searches past messages by meaning, not recency. Returns [{role, content}]."""
    if not query or not query.strip():
        return []

    count = _collection.count()
    if count == 0:
        return []

    embedding = _model.encode(query).tolist()
    results = _collection.query(query_embeddings=[embedding], n_results=min(n_results, count))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [{"role": m.get("role", "assistant"), "content": d} for d, m in zip(docs, metas)]