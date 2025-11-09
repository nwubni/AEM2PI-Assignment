import json
import numpy as np
import faiss
import argparse


def load_index(path):
    return faiss.read_index(path)


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def embed_query(query, dim):
    # NOTE: Replace this dummy embedding with actual model inference
    vec = np.random.randn(dim).astype(np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def parse_filter(filter_str):
    key, val = filter_str.split('=', 1)
    return key, val


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search: semantic + metadata filter")
    parser.add_argument('--query', type=str, default='example query')
    parser.add_argument('--index', choices=['flat', 'hnsw'], default='hnsw')
    parser.add_argument('--filter', type=str, required=True, help='Metadata filter, e.g., doc_type=whitepaper')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_candidates', type=int, default=50)
    args = parser.parse_args()

    index_path = 'hnsw_index.faiss' if args.index == 'hnsw' else 'flat_index.faiss'
    index = load_index(index_path)
    metadata = load_metadata('metadata_map.json')

    q_vec = embed_query(args.query, index.d)
    D, I = index.search(q_vec.reshape(1, -1), args.num_candidates)

    key, value = parse_filter(args.filter)
    results = []
    for dist, idx in zip(D[0], I[0]):
        md = metadata.get(str(idx))
        if md and md.get(key) == value:
            results.append((idx, dist, md))
        if len(results) == args.top_k:
            break

    print("Final Results:")
    for idx, dist, md in results:
        print(f"ID: {idx}, Score: {dist}, Metadata: {md}")


if __name__ == '__main__':
    main()