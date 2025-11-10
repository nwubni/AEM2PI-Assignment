"""
    This script takes in user input and builds a prompt for LLM to process
"""

import sys
import os
import json
import faiss
import numpy as np
from src.build_index import make_embedding
from model_api.endpoint import log_metrics, process_query


def load_faiss_index():
    """
    Load the faiss index from the file system
    """
    index_path = 'storage/index.faiss'
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        print(f"Warning: Index file not found at {index_path}. Please run build_index.py first.", file=sys.stderr)
        sys.exit(1)

def load_chunks():
    """
    Load the chunks from the file system
    """
    chunks_path = 'storage/chunks.json'
    if os.path.exists(chunks_path):
        return json.load(open(chunks_path))
    else:
        print(f"Warning: Chunks file not found at {chunks_path}. Please run build_index.py first.", file=sys.stderr)
        sys.exit(1)



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

        faq_chunks = [f"{chunks[str(faiss_id)]["text"]}\nSource: {chunks[str(faiss_id)]["document_id"]}" for faiss_id in I[0]]

        response, metrics = process_query(user_prompt, "\n".join(faq_chunks))

        if metrics.get("estimated_cost_usd", 0) > 0:
            metrics["actions"] = response["actions"]
            metrics["source"] = response["source"]
            metrics["user_question"] = user_prompt
            metrics["system_answer"] = response["answer"]
            metrics["chunks_related"] = faq_chunks
            log_metrics(metrics)

        print("\nMetrics:", json.dumps(metrics, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1) 

if __name__ == "__main__":
    main()