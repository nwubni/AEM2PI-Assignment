# AEM2PI-Assignment
Andela GenAI Second Assignment

This project uses Retrieval-Augmented Generation (RAG) to answer customer queries for an SAAS company. Customer queries are answered based on the company's FAQ document. It utilizes the LangChain library to achieve this objective because the library provides a high level of abstraction and ease of use for building production-ready RAG applications.

## Features

- LangChain-based RAG pipeline (FAISS vector store + HuggingFace embeddings + OpenAI LLM)
- Smart context truncation that preserves complete Q&A pairs to control token usage and cost
- Response caching with query normalization to reduce latency, quickly provide answers to repetitive queries, and lower API request costs
- Automated system answer quality evaluation
- Cost and latency metrics for every query, logged along side system responses to `outputs/sample_queries.json`
- Test suite with pytest for testing core functionalities

## Project Setup Instructions

1. Open your command terminal
2. Clone the repository in your directory of choice by running the git command:
   ```bash
   git clone https://github.com/nwubni/AEM2PI-Assignment.git
   ```
3. Change into the project's root directory:
   ```bash
   cd AEM2PI-Assignment
   ```
4. Create a virtual environment to isolate the project's dependencies:
   ```bash
   python3 -m venv .venv
   ```
5. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
6. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies:
   - `langchain`, `langchain-community`, `langchain-openai`, `langchain-huggingface`
   - `faiss-cpu` for vector storage and retrieval
   - `python-dotenv` for environment variables
   - `pytest`, `pytest-cov`, `pytest-mock` for tests
   - OpenAI models via `langchain-openai`

## Environment Variables

This project requires an OpenAI API key to function. Additionally, it requires the embedding model and LLM model to be specified. Create a `.env` file in the project root directory using the `.env.example` file for samples and add:
```
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-4o-mini
USE_CUDA=false
```

After the basic setup, the next step is to chunk the FAQ document, embed the chunks, and store them in a vector database. This project uses FAISS for vector storage and retrieval operations.
Run the following command to build the data pipeline.

`python -m src.build_index path/to/document.txt`

For this project, use this exact command:
```bash
python -m src.build_index data/faq_document.txt
```

## Query Pipeline
After building the data pipeline, you can send queries using this command format:

`python -m src.query "your question here"`

For example:
```bash
python -m src.query "What kind of things can you assist me with?"
```
System responses are logged to the console and also stored in `outputs/sample_queries.json`.

## Tests
This project includes a test suite to ensure it functions as intended.
Use the following command to run the tests:
```bash
pytest tests/test_core.py -v
```