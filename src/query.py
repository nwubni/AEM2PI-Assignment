"""
This script is used to build the query pipeline to answer user questions using the FAQ RAG system.
"""

from datetime import datetime, timezone
import json
import os
import sys
import time

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Import caching utilities
from scripts.cache import ResponseCache, normalize_query
from scripts.evaluator import AnswerEvaluator

# Load environment variables from .env file
load_dotenv()

# Model configuration
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1000000,  # $0.15 per 1M tokens input
        "output": 0.60 / 1000000,  # $0.60 per 1M tokens output
    }
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the cost of an API call based on token usage."""
    return round(
        (prompt_tokens * PRICING[model]["input"])
        + (completion_tokens * PRICING[model]["output"]),
        6,
    )


def get_relevant_documents(query: str, index_dir: str = "storage/faiss_index"):
    """
    Get relevant documents from the FAISS index.

    Args:
        query (str): The user query.
        index_dir (str, optional): Directory containing the FAISS index. Defaults to "storage/faiss_index".
    """

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        model_kwargs={"device": "cuda" if os.getenv("USE_CUDA") == "true" else "cpu"},
    )

    # Load vector store and get relevant documents
    vector_store = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(query, k=3)

    return docs


def load_prompt_template():
    """
    Load the main prompt template from file.

    Returns:
        str: The main prompt template.
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "main_prompt.txt"
    )
    with open(prompt_path, "r") as f:
        template = f.read()

    return template


def truncate_context(context_docs, max_tokens_per_chunk=400, max_total_chunks=3):
    """
    Truncate context for FAQ responses to only include Q&A pairs and discard any other content.
    
    Args:
        context_docs: List of retrieved documents
        max_tokens_per_chunk: Maximum tokens per chunk (default: 400)
        max_total_chunks: Maximum number of chunks to use (default: 3)
    
    Returns:
        str: Truncated context string
    """
    truncated_chunks = []
    
    for doc in context_docs[:max_total_chunks]:
        content = doc.page_content
        tokens = content.split()
        
        if len(tokens) > max_tokens_per_chunk:
            # Try to preserve complete Q&A format pairs
            if '\nA:' in content:
                # Find the last complete Q&A within token limit
                qa_pairs = content.split('\n\nQ:')
                truncated = qa_pairs[0]
                for qa in qa_pairs[1:]:
                    test_content = truncated + '\n\nQ:' + qa
                    if len(test_content.split()) <= max_tokens_per_chunk:
                        truncated = test_content
                    else:
                        break
                content = truncated
            else:
                # Do a simple truncation for non-Q&A formatted content
                content = ' '.join(tokens[:max_tokens_per_chunk])
        
        truncated_chunks.append(content)
    
    return "\n\n".join(truncated_chunks)


def log_metrics(output_data: dict, output_file: str = "outputs/sample_queries.json"):
    """
    Log query metrics to a JSON file.
    
    Args:
        output_data: Dictionary containing query metrics and results
        output_file: Path to the output JSON file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read existing data if file exists
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []

    # Append new result to existing metrics file
    existing_data.append(output_data)

    # Write back to file
    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=2)


def get_answer(query: str, context_docs, cache=None):
    """
    Process a user query and generate an answer using the LLM.

    Args:
        query (str): The user query.
        context_docs: The retrieved documents.
        cache: Optional ResponseCache instance for caching responses.
    """

    # Normalize query to improve cache hits
    normalized_query = normalize_query(query)

    # Check cache first
    if cache:
        cached_response = cache.get(normalized_query, context_docs)
        if cached_response:
            cached_response["cache_hit"] = True
            cached_response["latency_ms"] = 0  # 0 means it came from cache
            return cached_response
    
    # Apply truncation to context
    chunks_used = truncate_context(context_docs, max_tokens_per_chunk=400, max_total_chunks=3)

    # Load and format the prompt template
    template = load_prompt_template()
    system_prompt = template.replace("_QUESTION_ANSWER_", chunks_used)

    # Get the answer using the OpenAI API
    start_time = time.time()
    llm = ChatOpenAI(model=MODEL, temperature=0.7, max_tokens=150)

    # Create messages for ChatOpenAI using the system prompt and user query
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = llm.invoke(messages)
    end_time = time.time()

    # Calculate tokens (approximate - in production, use actual token counts from response)
    prompt_tokens = len(system_prompt.split()) + len(query.split())
    completion_tokens = len(response.content.split())

    # Parse the JSON response
    try:
        parsed_response = json.loads(response.content)
        result = {
            "answer": parsed_response.get("answer", response.content),
            "confidence": parsed_response.get("confidence", 0),
            "source": parsed_response.get("source", []),
            "actions": parsed_response.get("actions", []),
        }
    except json.JSONDecodeError:
        # If not valid JSON, use the raw response
        result = {
            "answer": response.content,
            "confidence": 0,
            "source": [],
            "actions": [],
        }

    response_data = {
        "answer": result.get("answer", "No answer provided"),
        "confidence": result.get("confidence", 0),
        "sources": result.get("source", []),
        "actions": result.get("actions", []),
        "tokens_prompt": prompt_tokens,
        "tokens_completion": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": (end_time - start_time) * 1000,
        "cache_hit": False,
        "chunks_related": [
            f"{doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}"
            for doc in context_docs
        ],
    }
    
    # Cache the response
    if cache:
        cache.set(normalized_query, context_docs, response_data)
    
    return response_data


def main():
    """Main entry point for the script."""

    if len(sys.argv) < 2:
        print('Usage: python -m src.query "Your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    try:
        # Initialize cache and evaluator
        cache = ResponseCache()
        evaluator = AnswerEvaluator()
        
        # Get relevant documents
        docs = get_relevant_documents(query)

        # Get answer and metadata with caching to improve latency and reduce cost
        result = get_answer(query, docs, cache=cache)

        # Calculate cost using the pricing model
        # If there is a cache hit, cost is 0
        if result.get("cache_hit", False):
            cost = 0.0
            print("\n✓ Response retrieved from cache (instant, $0 cost)")
        else:
            cost = calculate_cost(
                MODEL, result["tokens_prompt"], result["tokens_completion"]
            )
        
        # Evaluate the answer quality
        evaluation = evaluator.evaluate(
            query=query,
            answer=result["answer"],
            context_chunks=result["chunks_related"]
        )
        
        # Display quality scores
        print("\n" + "="*50)
        print("QUALITY ASSESSMENT")
        print("="*50)
        print(evaluator.get_quality_summary(evaluation))

        # Prepare metrics
        output_data = {
            "model": MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tokens_prompt": result["tokens_prompt"],
            "tokens_completion": result["tokens_completion"],
            "total_tokens": result["total_tokens"],
            "latency_ms": round(result["latency_ms"], 2),
            "estimated_cost_usd": cost,
            "cache_hit": result.get("cache_hit", False),
            "actions": result["actions"],
            "source": result["sources"],
            "user_question": query,
            "system_answer": result["answer"],
            "chunks_related": result["chunks_related"],
            "confidence": result["confidence"],
            "evaluation": {
                "overall_score": evaluation.get("overall_score", 0),
                "quality_level": evaluation.get("quality_level", "unknown"),
                "relevance_score": evaluation.get("relevance", {}).get("score", 0),
                "accuracy_score": evaluation.get("accuracy", {}).get("score", 0),
                "completeness_score": evaluation.get("completeness", {}).get("score", 0),
                "clarity_score": evaluation.get("clarity", {}).get("score", 0),
                "improvements": evaluation.get("improvements", [])
            }
        }

        # Save metrics to JSON file
        log_metrics(output_data)

        # Print results
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Source {i} ---")
            print(
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )

        print(f"\nResults saved to outputs/sample_queries.json")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
