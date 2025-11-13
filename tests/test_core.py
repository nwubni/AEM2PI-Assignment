"""
Test suite for RAG FAQ system.
Tests cover caching, query processing, document retrieval, and answer quality.
"""

import os
from unittest.mock import Mock

import pytest

from scripts.cache import ResponseCache, normalize_query
from scripts.evaluator import AnswerEvaluator
from src.query import get_answer, get_relevant_documents, log_metrics, truncate_context


class TestQueryNormalization:
    """Test query normalization for better cache hits."""

    def test_lowercase_conversion(self):
        """Test that queries are converted to lowercase."""
        query = "How Do I Reset My Password?"
        normalized = normalize_query(query)
        assert normalized == "how can i reset my password"

    def test_whitespace_removal(self):
        """Test that extra whitespace is removed."""
        query = "How   do   I    reset   password?"
        normalized = normalize_query(query)
        assert "  " not in normalized

    def test_punctuation_removal(self):
        """Test that trailing punctuation is removed."""
        assert normalize_query("What is shipping?") == "what is shipping"
        assert normalize_query("What is shipping!") == "what is shipping"
        assert normalize_query("What is shipping.") == "what is shipping"

    def test_common_variations(self):
        """Test that common variations are normalized."""
        assert "how can i" in normalize_query("How do I reset password?")
        assert "what is" in normalize_query("What's the shipping cost?")
        assert "cannot" in normalize_query("I can't find my order")


class TestCaching:
    """Test response caching functionality."""

    def setup_method(self):
        """Setup test cache with temporary file."""
        self.cache_file = "storage/test_cache.json"
        self.cache = ResponseCache(cache_file=self.cache_file)

    def teardown_method(self):
        """Clean up test cache file."""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    def test_cache_miss(self):
        """Test that new queries result in cache miss."""
        mock_docs = [Mock(page_content="test content")]
        result = self.cache.get("test query", mock_docs)
        assert result is None

    def test_cache_hit(self):
        """Test that cached queries return stored response."""
        mock_docs = [Mock(page_content="test content")]
        test_response = {
            "answer": "Test answer",
            "confidence": 90,
            "cache_hit": False
        }
        
        # Cache the response
        self.cache.set("test query", mock_docs, test_response)
        
        # Retrieve from cache
        cached = self.cache.get("test query", mock_docs)
        assert cached is not None
        assert cached["answer"] == "Test answer"
        assert cached["confidence"] == 90

    def test_cache_persistence(self):
        """Test that cache persists to disk."""
        mock_docs = [Mock(page_content="test content")]
        test_response = {"answer": "Test", "confidence": 100}
        
        self.cache.set("test query", mock_docs, test_response)
        
        # Create new cache instance
        new_cache = ResponseCache(cache_file=self.cache_file)
        cached = new_cache.get("test query", mock_docs)
        assert cached is not None

    def test_cache_key_uniqueness(self):
        """Test that different queries generate different cache keys."""
        key1 = self.cache.get_cache_key("query 1", "hash1")
        key2 = self.cache.get_cache_key("query 2", "hash2")
        
        assert key1 != key2


class TestContextTruncation:
    """Test smart context truncation."""

    def test_preserves_qa_pairs(self):
        """Test that Q&A pairs are preserved during truncation."""
        mock_doc = Mock(
            page_content="Q: Test question?\nA: Test answer.\n\nQ: Another question?\nA: Another answer."
        )
        
        result = truncate_context([mock_doc], max_tokens_per_chunk=50, max_total_chunks=1)
        
        assert "Q:" in result
        assert "A:" in result

    def test_respects_chunk_limit(self):
        """Test that truncation respects max chunk count."""
        mock_docs = [
            Mock(page_content=f"Content {i}") for i in range(10)
        ]
        
        result = truncate_context(mock_docs, max_total_chunks=3)
        chunks = result.split("\n\n")
        
        assert len(chunks) <= 3

    def test_respects_token_limit(self):
        """Test that truncation respects token limit per chunk."""
        long_content = " ".join(["word"] * 1000)
        mock_doc = Mock(page_content=long_content)
        
        result = truncate_context([mock_doc], max_tokens_per_chunk=100)
        token_count = len(result.split())
        
        assert token_count <= 100


class TestDocumentRetrieval:
    """Test document retrieval from FAISS index."""

    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_retrieval_returns_documents(self):
        """Test that retrieval returns documents."""
        query = "How do I reset my password?"
        docs = get_relevant_documents(query)
        
        assert len(docs) > 0
        assert len(docs) <= 3

    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_retrieval_relevance(self):
        """Test that retrieved documents are relevant."""
        query = "How do I reset my password?"
        docs = get_relevant_documents(query)
        
        # Check if any document contains password-related content
        content = " ".join([doc.page_content for doc in docs])
        assert "password" in content.lower()

    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_retrieval_has_metadata(self):
        """Test that retrieved documents have metadata."""
        query = "What is the company's remote work policy?"
        docs = get_relevant_documents(query)
        
        for doc in docs:
            assert hasattr(doc, 'metadata')
            assert 'source' in doc.metadata


class TestFAQQueries:
    """Test FAQ-specific queries."""

    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_password_reset_query(self):
        """Test password reset FAQ query."""
        query = "How do I reset my password?"
        docs = get_relevant_documents(query)
        
        content = " ".join([doc.page_content for doc in docs])
        assert "password" in content.lower()
        assert any(word in content for word in ["Forgot", "reset", "email"])

    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_remote_work_query(self):
        """Test remote work policy FAQ query."""
        query = "What is the company's remote work policy?"
        docs = get_relevant_documents(query)

        content = " ".join([doc.page_content for doc in docs])
        assert "remote" in content.lower()

    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_expense_reimbursement_query(self):
        """Test expense reimbursement policy FAQ query."""
        query = "What is the company's expense reimbursement policy?"
        docs = get_relevant_documents(query)

        content = " ".join([doc.page_content for doc in docs])
        assert ("expense" in content.lower()) or ("reimbursement" in content.lower())


class TestAnswerGeneration:
    """Test answer generation with LLM."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OpenAI API key not set"
    )
    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_answer_structure(self):
        """Test that answer has correct structure."""
        query = "How do I reset my password?"
        docs = get_relevant_documents(query)
        cache = ResponseCache(cache_file="storage/test_answer_cache.json")
        
        result = get_answer(query, docs, cache=cache)
        
        # Check structure
        assert "answer" in result
        assert "confidence" in result
        assert "sources" in result
        assert "actions" in result
        assert "cache_hit" in result
        
        # Clean up
        if os.path.exists("storage/test_answer_cache.json"):
            os.remove("storage/test_answer_cache.json")

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OpenAI API key not set"
    )
    @pytest.mark.skipif(
        not os.path.exists("storage/faiss_index"),
        reason="FAISS index not built"
    )
    def test_cache_reduces_latency(self):
        """Test that cache significantly reduces latency."""
        query = "What are your shipping options?"
        docs = get_relevant_documents(query)
        cache = ResponseCache(cache_file="storage/test_latency_cache.json")
        
        # First call (cache miss)
        result1 = get_answer(query, docs, cache=cache)
        latency1 = result1["latency_ms"]
        
        # Second call (cache hit)
        result2 = get_answer(query, docs, cache=cache)
        latency2 = result2["latency_ms"]
        
        assert result2["cache_hit"] is True
        assert latency2 < 100  # Cache should be very fast
        assert latency2 < latency1  # Cache should be faster
        
        # Clean up
        if os.path.exists("storage/test_latency_cache.json"):
            os.remove("storage/test_latency_cache.json")


class TestEvaluator:
    """Test answer quality evaluator."""

    def test_evaluator_initialization(self):
        """Test that evaluator initializes correctly."""
        evaluator = AnswerEvaluator()
        assert evaluator.model is not None
        assert evaluator.llm is not None

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OpenAI API key not set"
    )
    def test_evaluation_structure(self):
        """Test that evaluation has correct structure."""
        evaluator = AnswerEvaluator()
        
        query = "How do I reset my password?"
        answer = "Click on 'Log In' and then 'Forgot Password'."
        context = ["Q: How do I reset my password?\nA: Click on 'Log In' and then 'Forgot Password'."]
        
        evaluation = evaluator.evaluate(query, answer, context)
        
        # Check structure
        assert "overall_score" in evaluation
        assert "quality_level" in evaluation
        assert "relevance" in evaluation or "relevance_score" in evaluation
        assert "accuracy" in evaluation or "accuracy_score" in evaluation

    def test_quality_threshold(self):
        """Test quality threshold logic."""
        evaluator = AnswerEvaluator()
        
        good_eval = {"overall_score": 8.5}
        poor_eval = {"overall_score": 4.0}
        
        assert evaluator.should_show_to_user(good_eval, threshold=6.0) is True
        assert evaluator.should_show_to_user(poor_eval, threshold=6.0) is False


class TestMetricsLogging:
    """Test metrics logging functionality."""

    def test_log_metrics_creates_file(self):
        """Test that log_metrics creates output file."""
        test_output = "outputs/test_metrics.json"
        test_data = {
            "model": "gpt-4o-mini",
            "user_question": "test",
            "system_answer": "test answer"
        }
        
        log_metrics(test_data, output_file=test_output)
        
        assert os.path.exists(test_output)
        
        # Clean up
        if os.path.exists(test_output):
            os.remove(test_output)

    def test_log_metrics_appends_data(self):
        """Test that log_metrics appends to existing file."""
        import json
        
        test_output = "outputs/test_append.json"
        
        # Log first entry
        log_metrics({"query": "first"}, output_file=test_output)
        
        # Log second entry
        log_metrics({"query": "second"}, output_file=test_output)
        
        # Read and verify
        with open(test_output, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]["query"] == "first"
        assert data[1]["query"] == "second"
        
        # Clean up
        if os.path.exists(test_output):
            os.remove(test_output)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query_normalization(self):
        """Test normalization of empty query."""
        result = normalize_query("")
        assert result == ""

    def test_very_long_query_normalization(self):
        """Test normalization of very long query."""
        long_query = "word " * 1000
        result = normalize_query(long_query)
        assert isinstance(result, str)

    def test_special_characters_normalization(self):
        """Test normalization with special characters."""
        query = "What's the @#$% shipping cost?"
        result = normalize_query(query)
        assert isinstance(result, str)
        assert "what is" in result
