"""
tools/rag_tools.py
ChromaDB RAG tools for the Diagnoser and Solution agents.

Two collections:
- "incidents" : past IT incident reports (for diagnosis)
- "playbooks" : fix procedures and runbooks (for solutions)
"""

from __future__ import annotations
from typing import Optional, List, Dict

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from config.settings import CHROMA_DB_PATH, EMBEDDING_MODEL


# ── Singleton ChromaDB client ─────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_ef = None


def _get_client():
    global _client, _ef
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            # Quick test to ensure it works
            _ef(["test"])
            logger.info(f"[CHROMA] Using SentenceTransformer ({EMBEDDING_MODEL})")
        except Exception as e:
            logger.warning(f"[CHROMA] SentenceTransformer failed ({e}), using default embeddings")
            _ef = embedding_functions.DefaultEmbeddingFunction()
        logger.info(f"[CHROMA] Connected to store at {CHROMA_DB_PATH}")
    return _client, _ef


def _get_collection(collection_name: str = "incidents"):
    client, ef = _get_client()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── Query KB ──────────────────────────────────────────────────────────────────

def query_kb(
    query_text: str,
    n_results: int = 3,
    collection_name: str = "incidents",
) -> List[Dict]:
    """
    Query ChromaDB for documents similar to query_text.

    Args:
        query_text: The text to search for
        n_results: Number of results to return
        collection_name: "incidents" or "playbooks"

    Returns:
        List of {document, metadata, distance} dicts
    """
    try:
        collection = _get_collection(collection_name)
        count = collection.count()
        if count == 0:
            logger.warning(f"[RAG] Collection '{collection_name}' is empty. Run seed_kb.py first.")
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        logger.debug(f"[RAG] Query: '{query_text[:50]}' → {len(docs)} results from '{collection_name}'")
        return docs

    except Exception as e:
        logger.error(f"[RAG] Query failed: {e}")
        return []


# ── Upsert documents ──────────────────────────────────────────────────────────

def upsert_document(
    doc_id: str,
    text: str,
    metadata: dict,
    collection_name: str = "incidents",
) -> bool:
    """
    Add or update a document in ChromaDB.

    Args:
        doc_id: Unique document ID
        text: Document text content
        metadata: Dict of metadata (category, source, etc.)
        collection_name: "incidents" or "playbooks"

    Returns:
        True on success
    """
    try:
        collection = _get_collection(collection_name)
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
        )
        logger.debug(f"[RAG] Upserted doc '{doc_id}' into '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"[RAG] Upsert failed: {e}")
        return False


def upsert_batch(
    documents: List[Dict],
    collection_name: str = "incidents",
) -> int:
    """
    Batch upsert documents.

    Args:
        documents: List of {id, text, metadata} dicts
        collection_name: Target collection

    Returns:
        Number of documents upserted
    """
    try:
        collection = _get_collection(collection_name)
        collection.upsert(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            metadatas=[d["metadata"] for d in documents],
        )
        logger.info(f"[RAG] Batch upserted {len(documents)} docs into '{collection_name}'")
        return len(documents)
    except Exception as e:
        logger.error(f"[RAG] Batch upsert failed: {e}")
        return 0


def get_collection_stats() -> dict:
    """Return count of documents in each collection."""
    stats = {}
    for name in ["incidents", "playbooks"]:
        try:
            col = _get_collection(name)
            stats[name] = col.count()
        except Exception:
            stats[name] = 0
    return stats
