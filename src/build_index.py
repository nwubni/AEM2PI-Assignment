"""
    This script is used to build an index of a document.
    It chunks the document into smaller texts, 
    generates embeddings for each chunk, and saves the embeddings to a file.
"""

import datetime
import json
import os
import sys
import uuid

import dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

dotenv.load_dotenv()

DIM = 384
index = faiss.IndexFlatL2(DIM)

def load_document(path:str) -> str:
    """
    Arguments:
        path: File path of document to load

    Returns:
        A string of the document
    """

    with open(path, 'r') as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 40, overlap_size: int = 2) -> [str]:
    """
    Arguments:
        text: Input text to chunk
        chunk_size: Specifies the number of chunks with default value of 40
        overlap_size: Specifies the number of words to overlap with default value of 2
    Returns:
        A list of chunked texts
    """

    # Takes text and chunks it into smaller texts
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunks.append(' '.join(words[start:end]))

        start += chunk_size - overlap_size

    return chunks


def create_chunks_with_metadata(document_id, document_text, chunk_fn=chunk_text):
    """
    Arguments:
        document_id: The id of the document
        document_text: The text of the document
        chunk_fn: The function to use to chunk the document
    Returns:
        A list of chunk objects
    """

    chunks = chunk_fn(document_text)
    chunk_objects = []   

    for idx, chunk_text in enumerate(chunks):
        chunk_obj = {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "chunk_index": idx,
            "text": chunk_text,
            "char_start": document_text.find(chunk_text),
            "char_end": document_text.find(chunk_text) + len(chunk_text),
            "token_count": len(chunk_text.split()),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat()
       }

        chunk_objects.append(chunk_obj)

    return chunk_objects

def save_chunks_metadata(metadata_map: dict, path: str):
    """
    Arguments:
        metadata_map: A dictionary of metadata map
        path: File path to store the chunks
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(metadata_map, f)

def generate_embeddings(chunks:[dict], batch_size: int = 16):
    """
        Arguments:
            chunks: A list of chunked texts

    Returns:
        A list of embeddings, each embedding is a dictionary with the following keys:
            - vector: A list of floats
            - metadata: A dictionary with the following keys:
                - id: A string
                - doc_type: A string
                - department: A string
    """

    # Takes chunked texts and generates their embeddings
    texts = [chunk["text"] for chunk in chunks]

   # Use GPU if available

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(os.getenv('EMBEDDING_MODEL')).to(device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )

    return embeddings.tolist()


def make_embedding(text: str) -> np.array:
    """
    Arguments:
        text: A string to generate an embedding for
    Returns:
        A numpy array of the embedding
    """
    return generate_embeddings([{"text": text}])[0]

def load_embeddings(path: str) -> [dict]:
    """
    Arguments:
        path: File path to load the embeddings from
    Returns:
        A list of embeddings, each embedding is a dictionary with the following keys:
            - vector: A list of floats
            - metadata: A dictionary with the following keys:
                - id: A string

                - doc_type: A string
                - department: A string
    """
    # Takes a file path and loads the embeddings from it
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def main():
    """Main entry point for the build script."""
    if len(sys.argv) < 2:
        print('Usage: python -m src.build_index "path/to/document.txt"')
        sys.exit(1)
    document_path = sys.argv[1]

    text = load_document(document_path)
    chunks = create_chunks_with_metadata(document_id="faq_document", document_text=text)
    embeddings = generate_embeddings(chunks)

    for chunk, embedding_vector in zip(chunks, embeddings):
        chunk["embedding"] = embedding_vector

    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)

    start_id = index.ntotal
    index.add(vectors)

    metadata_map = {}

    for offset, chunk in enumerate(chunks):
        metadata_map[start_id + offset] = chunk

    os.makedirs('outputs', exist_ok=True)
    faiss.write_index(index, 'outputs/index.faiss')

    save_chunks_metadata(metadata_map, 'outputs/chunks.json')

if __name__ == "__main__":
    main()