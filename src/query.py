"""
    This script takes in user input and builds a prompt for LLM to process
"""

import sys
import os
import json
import faiss
import numpy as np
from src.build_index import make_embedding


def load_faiss_index():
    """
    Load the faiss index from the file system
    """
    index_path = 'outputs/index.faiss'
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        print(f"Warning: Index file not found at {index_path}. Please run build_index.py first.", file=sys.stderr)
        sys.exit(1)

def load_chunks():
    """
    Load the chunks from the file system
    """
    chunks_path = 'outputs/chunks.json'
    if os.path.exists(chunks_path):
        return json.load(open(chunks_path))
    else:
        print(f"Warning: Chunks file not found at {chunks_path}. Please run build_index.py first.", file=sys.stderr)
        sys.exit(1)


def hybrid_search(embedded_user_prompt, embeddings):
    """
    Arguments:
        embedded_user_prompt: An embedded user prompt
        embeddings: A list of embeddings
    Returns:
        A list of results
    """
    pass

def query_index(embedded_user_prompt, embeddings):

    """
    Arguments:
        embedded_user_prompt: An embedded user prompt
        embeddings: A list of embeddings
    Returns:
        A list of results
    """
    pass


index = load_faiss_index()
chunks = load_chunks()

def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "Your question here"')
        sys.exit(1)

    user_prompt = sys.argv[1]

    try:
        query_vector = np.array(make_embedding(user_prompt), dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        D, I = index.search(query_vector, k=3)


        for faiss_id in I[0]:
            print(chunks[str(faiss_id)]["text"])

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1) 

if __name__ == "__main__":
    main()