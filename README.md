# AEM2PI-Assignment
Andela GenAI Second Assignment

This project uses RAG to answer customer queries for an ecommerce retail store.

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
   
   The dependencies include OpenAI to use OpenAI models to process user prompts. It also installs the sentence transformers library for text embedding generations, and FAISS for vector storage and retrieval operations.

## Environment Variables

This project requires an OpenAI API key to function. Additionally, it requires the vector dimension and embedding model to be specified. Create a `.env` file in the project root directory using the `.env.example` file for samples and add:
```
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=all-minilm-l6-v2
VECTOR_DIM=384
```

After the basic setup, the next step is to chunk the FAQ document, embbed the chunks and store them in a vector database. This project uses FAISS for vector storage and retrieval operations.
Run the following command to build the data pipeline.

`python -m src.build_index path_to_document`

For this project, use this exact command:
```bash
python -m src.build_index data/faq_document.txt
```


Query Pipeline
After building the data pipeline, users can send their queries using this command format `python -m src.query "user_query"`

For example:
```bash
python -m src.query "What kind of things can you assist me with?"
```