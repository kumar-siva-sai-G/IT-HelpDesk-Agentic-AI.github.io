from __future__ import annotations
from typing import List, Dict, Optional
"""
memory/mem0_client.py
Mem0 wrapper for persistent cross-session agent memory.

Falls back to local JSON storage if MEM0_API_KEY is not set.
"""

import json
import os
import time
from pathlib import Path
from loguru import logger

from config.settings import MEM0_API_KEY, BASE_DIR


class Mem0Client:
    """
    Unified memory client that uses Mem0 cloud if API key is present,
    otherwise falls back to a local JSON file store.
    """

    def __init__(self):
        self._use_cloud = bool(MEM0_API_KEY)
        self._local_path = BASE_DIR / "memory" / "local_memory.json"
        self._local_path.parent.mkdir(exist_ok=True)

        if self._use_cloud:
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=MEM0_API_KEY)
                logger.info("[MEM0] Using Mem0 cloud storage")
            except ImportError:
                logger.warning("[MEM0] mem0ai not installed. Falling back to local storage.")
                self._use_cloud = False
                self._client = None
        else:
            logger.info("[MEM0] Using local JSON memory store")
            self._client = None

    def store_memory(self, user_id: str, text: str, metadata: dict = None) -> bool:
        """Store a memory entry for a user."""
        if self._use_cloud and self._client:
            try:
                self._client.add(
                    [{"role": "user", "content": text}],
                    user_id=user_id,
                    metadata=metadata or {},
                )
                logger.debug(f"[MEM0] Stored memory for user {user_id}")
                return True
            except Exception as e:
                logger.error(f"[MEM0] Cloud store error: {e}")
                return False
        else:
            return self._local_store(user_id, text, metadata)

    def get_memories(self, user_id: str, query: str = "", limit: int = 5) -> List[Dict]:
        """Retrieve memories for a user, optionally filtered by query."""
        if self._use_cloud and self._client:
            try:
                results = self._client.search(
                    query=query or "IT ticket history",
                    filters={"user_id": user_id},
                    limit=limit,
                )
                return results if isinstance(results, list) else results.get("results", [])
            except Exception as e:
                logger.error(f"[MEM0] Cloud search error: {e}")
                return []
        else:
            return self._local_search(user_id, query, limit)

    def _local_store(self, user_id: str, text: str, metadata: dict = None) -> bool:
        """Local JSON fallback for storing memories."""
        try:
            store = self._load_local()
            if user_id not in store:
                store[user_id] = []
            store[user_id].append({
                "memory": text,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })
            # Keep only last 50 memories per user
            store[user_id] = store[user_id][-50:]
            self._save_local(store)
            return True
        except Exception as e:
            logger.error(f"[MEM0] Local store error: {e}")
            return False

    def _local_search(self, user_id: str, query: str, limit: int) -> List[Dict]:
        """Local JSON fallback for searching memories."""
        try:
            store = self._load_local()
            user_memories = store.get(user_id, [])
            if not query:
                return user_memories[-limit:]

            # Simple keyword search
            query_words = query.lower().split()
            scored = []
            for mem in user_memories:
                text = mem.get("memory", "").lower()
                score = sum(1 for w in query_words if w in text)
                if score > 0:
                    scored.append((score, mem))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:limit]]

        except Exception as e:
            logger.error(f"[MEM0] Local search error: {e}")
            return []

    def _load_local(self) -> dict:
        if self._local_path.exists():
            with open(self._local_path) as f:
                return json.load(f)
        return {}

    def _save_local(self, store: dict):
        with open(self._local_path, "w") as f:
            json.dump(store, f, indent=2)
