"""
Response caching module for FAQ RAG system.
Provides caching functionality to reduce API costs and improve response times.
"""

from datetime import datetime, timezone
import hashlib
import json
import os


class ResponseCache:
    """Cache for storing and retrieving LLM responses."""

    def __init__(self, cache_file="storage/response_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_cache(self):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def get_cache_key(self, query, context_hash):
        """Generate cache key from query and context."""
        combined = f"{query.lower().strip()}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, query, context_docs):
        """Get cached response if available."""
        context_hash = hashlib.md5(
            "".join([doc.page_content for doc in context_docs]).encode()
        ).hexdigest()[:8]

        cache_key = self.get_cache_key(query, context_hash)
        cached = self.cache.get(cache_key)

        if cached:
            return cached.get("response")

        return None

    def set(self, query, context_docs, response):
        """Cache a response."""
        context_hash = hashlib.md5(
            "".join([doc.page_content for doc in context_docs]).encode()
        ).hexdigest()[:8]

        cache_key = self.get_cache_key(query, context_hash)
        self.cache[cache_key] = {
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
        }
        self._save_cache()

    def clear(self):
        """Clear all cached responses."""
        self.cache = {}
        self._save_cache()

    def get_stats(self):
        """Get cache statistics."""
        return {"total_entries": len(self.cache), "cache_file": self.cache_file}


def normalize_query(query):
    """
    Normalize query for better cache hit rates.

    Args:
        query (str): The user query.

    Returns:
        str: Normalized query.
    """

    query = query.lower().strip()
    query = " ".join(query.split())
    query = query.rstrip("?!.")

    # Common variations
    replacements = {
        "how do i": "how can i",
        "what's": "what is",
        "can't": "cannot",
        "won't": "will not",
        "i'm": "i am",
        "you're": "you are",
    }
    for old, new in replacements.items():
        query = query.replace(old, new)

    return query
