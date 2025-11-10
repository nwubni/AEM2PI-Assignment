"""
Execute a retrieval query with caching and deduplication.
"""
import argparse, hashlib
from utils.cache import Cache
from retrieve_module import retrieve

CACHE = Cache()

def fingerprint(query, k, weights):
    s = f"{query}|{k}|{weights}"
    return hashlib.md5(s.encode()).hexdigest()

def dedupe(chunks):
    seen = set()
    unique = []
    for c in chunks:
        h = hashlib.md5(c['text'].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    removed = len(chunks) - len(unique)
    print(f"Removed {removed} duplicate chunks")
    return unique