import json
import numpy as np
import faiss

EMBEDDING_FILE = 'sample_embeddings.jsonl'
DIM = 128


def main():
    vectors = []
    metadata_map = {}

    with open(EMBEDDING_FILE, 'r') as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            vec = np.array(obj['vector'], dtype=np.float32)
            if vec.shape[0] != DIM:
                raise ValueError(f"Vector dimension mismatch at line {idx}")
            faiss.normalize_L2(vec.reshape(1, -1))
            vectors.append(vec)
            metadata_map[idx] = obj['metadata']

    xb = np.vstack(vectors)

    # Exact (brute-force) index
    index_flat = faiss.IndexFlatIP(DIM)
    index_flat.add(xb)
    faiss.write_index(index_flat, 'flat_index.faiss')

    # HNSW index for approximate search
    index_hnsw = faiss.IndexHNSWFlat(DIM, 32)
    index_hnsw.hnsw.efConstruction = 40
    index_hnsw.add(xb)
    faiss.write_index(index_hnsw, 'hnsw_index.faiss')

    # Save metadata map
    with open('metadata_map.json', 'w') as f:
        json.dump(metadata_map, f)

    print("Indexes and metadata map saved.")


if __name__ == '__main__':
    main()