import json
import random
import numpy as np

NUM_VECTORS = 10000
DIMENSION = 128
DOC_TYPES = ['report', 'article', 'whitepaper', 'blog_post']
DEPARTMENTS = ['finance', 'engineering', 'marketing', 'sales']


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def main():
    with open('sample_embeddings.jsonl', 'w') as f:
        for i in range(NUM_VECTORS):
            vec = np.random.randn(DIMENSION).astype(np.float32)
            vec = normalize(vec).tolist()
            metadata = {
                'id': f'doc_{i}',
                'doc_type': random.choice(DOC_TYPES),
                'department': random.choice(DEPARTMENTS)
            }
            record = {'vector': vec, 'metadata': metadata}
            f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()