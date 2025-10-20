"""Unit tests for RetrievalService."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.services.retrieval_service import RetrievalService
from app.models import SearchQuery, SearchResult, SearchResponse


@pytest.mark.unit
class TestRetrievalService:
    """Test cases for RetrievalService."""

    @pytest.fixture
    def retrieval_service(self):
        """Create RetrievalService instance with mocked dependencies."""
        with patch('app.services.retrieval_service.ElasticsearchService') as mock_es, \
             patch('app.services.retrieval_service.EmbeddingService') as mock_embedding:
            
            service = RetrievalService()
            service.es_service = mock_es.return_value
            service.embedding_service = mock_embedding.return_value
            return service

    @pytest.mark.asyncio
    async def test_search_success(self, retrieval_service, sample_search_query, sample_search_results):
        """Test successful search operation."""
        # Mock embedding generation
        retrieval_service.embedding_service.generate_query_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )
        
        # Mock Elasticsearch search
        retrieval_service.es_service.hybrid_search = AsyncMock(
            return_value=sample_search_results
        )
        
        result = await retrieval_service.search(sample_search_query)
        
        assert isinstance(result, SearchResponse)
        assert result.query == sample_search_query.query
        assert len(result.results) == len(sample_search_results)
        assert result.total_results == len(sample_search_results)
        assert result.processing_time_ms > 0
        # Reranking only happens when there are more results than final_top_k (10)
        # Since we only have 2 results, reranking won't be triggered
        assert result.used_reranking == False  # Not enough results for reranking
        # Compression is currently disabled in the implementation
        assert result.used_compression == False  # Compression is not implemented yet

    @pytest.mark.asyncio
    async def test_search_with_reranking(self, retrieval_service, sample_search_query, sample_search_results):
        """Test search with reranking enabled."""
        # Mock embedding generation
        retrieval_service.embedding_service.generate_query_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )
        
        # Mock Elasticsearch search with more results for reranking (need >10 for reranking to trigger)
        extended_results = sample_search_results + [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"Additional chunk {i} about various topics",
                score=0.9 - (i * 0.05),
                document_id="doc_1",
                level=0,
                rank=i+3,
                relevance_score=0.9 - (i * 0.05),
                token_count=8
            ) for i in range(3, 15)  # Add 12 more results (total 14 > 10)
        ]
        retrieval_service.es_service.hybrid_search = AsyncMock(
            return_value=extended_results
        )
        
        # Mock reranking
        with patch.object(retrieval_service, '_rerank_results', new_callable=AsyncMock) as mock_rerank:
            mock_rerank.return_value = sample_search_results
            
            result = await retrieval_service.search(sample_search_query)
            
            assert result.used_reranking is True
            mock_rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_compression(self, retrieval_service, sample_search_query, sample_search_results):
        """Test search with compression enabled (currently disabled in implementation)."""
        # Mock embedding generation
        retrieval_service.embedding_service.generate_query_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )
        
        # Mock Elasticsearch search
        retrieval_service.es_service.hybrid_search = AsyncMock(
            return_value=sample_search_results
        )
        
        result = await retrieval_service.search(sample_search_query)
        
        # Compression is currently disabled in the implementation
        assert result.used_compression is False

    @pytest.mark.asyncio
    async def test_search_with_filters(self, retrieval_service, sample_search_query, sample_search_results):
        """Test search with filters applied."""
        # Mock embedding generation
        retrieval_service.embedding_service.generate_query_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )
        
        # Mock Elasticsearch search
        retrieval_service.es_service.hybrid_search = AsyncMock(
            return_value=sample_search_results
        )
        
        # Set filters
        sample_search_query.filter_by_part = 1
        sample_search_query.filter_by_chapter = 2
        
        result = await retrieval_service.search(sample_search_query)
        
        # Verify filters were passed to Elasticsearch
        retrieval_service.es_service.hybrid_search.assert_called_once()
        call_args = retrieval_service.es_service.hybrid_search.call_args
        filters = call_args[1]["filters"]
        assert filters["part_number"] == 1
        assert filters["chapter_number"] == 2

    @pytest.mark.asyncio
    async def test_search_error(self, retrieval_service, sample_search_query):
        """Test search error handling."""
        # Mock embedding generation to raise error
        retrieval_service.embedding_service.generate_query_embedding = AsyncMock(
            side_effect=Exception("Embedding generation failed")
        )
        
        with pytest.raises(Exception):
            await retrieval_service.search(sample_search_query)

    @pytest.mark.asyncio
    async def test_rerank_results(self, retrieval_service, sample_search_results):
        """Test result reranking."""
        query = "Rome Italy travel"
        
        result = await retrieval_service._rerank_results(query, sample_search_results)
        
        assert len(result) == len(sample_search_results)
        # Results should be sorted by score (highest first)
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    @pytest.mark.asyncio
    async def test_rerank_results_error(self, retrieval_service, sample_search_results):
        """Test reranking error handling."""
        query = "Rome Italy travel"
        
        # Mock the rerank score calculation to raise an error
        with patch.object(retrieval_service, '_calculate_rerank_score', side_effect=Exception("Rerank error")):
            result = await retrieval_service._rerank_results(query, sample_search_results)
            
            # Should return original results on error
            assert result == sample_search_results

    def test_calculate_rerank_score(self, retrieval_service):
        """Test rerank score calculation."""
        query = "Rome Italy travel"
        text = "This is about Rome and Italy travel destinations."
        
        score = retrieval_service._calculate_rerank_score(query, text)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have some overlap

    def test_calculate_rerank_score_no_overlap(self, retrieval_service):
        """Test rerank score calculation with no overlap."""
        query = "Paris France"
        text = "This is about Rome and Italy travel destinations."
        
        score = retrieval_service._calculate_rerank_score(query, text)
        
        assert score == 0.0

    def test_calculate_rerank_score_empty_query(self, retrieval_service):
        """Test rerank score calculation with empty query."""
        query = ""
        text = "This is about Rome and Italy travel destinations."
        
        score = retrieval_service._calculate_rerank_score(query, text)
        
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_compress_results(self, retrieval_service, sample_search_results):
        """Test result compression."""
        query = "Rome Italy travel"
        
        # Mock settings for compression
        with patch('app.services.retrieval_service.settings') as mock_settings:
            mock_settings.max_tokens_compression = 50  # Low limit to trigger compression
            
            result = await retrieval_service._compress_results(query, sample_search_results)
            
            assert len(result) <= len(sample_search_results)
            # Compressed results should have shorter text
            for i, res in enumerate(result):
                if i < len(sample_search_results):
                    assert len(res.text) <= len(sample_search_results[i].text)

    @pytest.mark.asyncio
    async def test_compress_results_error(self, retrieval_service, sample_search_results):
        """Test compression error handling."""
        query = "Rome Italy travel"
        
        # Mock the compression to raise an error
        with patch.object(retrieval_service, '_compress_text', side_effect=Exception("Compression error")):
            result = await retrieval_service._compress_results(query, sample_search_results)
            
            # Should return original results on error
            assert result == sample_search_results

    def test_compress_text(self, retrieval_service):
        """Test text compression."""
        query = "Rome Italy travel"
        text = "This is a long text about Rome and Italy travel destinations. It contains multiple sentences about various places to visit in Italy. Rome is the capital and has many historical sites. Venice is known for its canals. Florence has beautiful art and architecture."
        
        compressed = retrieval_service._compress_text(query, text)
        
        assert len(compressed) <= len(text)
        assert "Rome" in compressed or "Italy" in compressed  # Should keep relevant parts

    def test_compress_text_short(self, retrieval_service):
        """Test text compression with short text."""
        query = "Rome Italy travel"
        text = "Short text about Rome."
        
        compressed = retrieval_service._compress_text(query, text)
        
        # The method always compresses, even short text
        assert len(compressed) <= len(text)
        # The compression method may truncate text, so just check it's shorter
        assert isinstance(compressed, str)

    def test_estimate_tokens(self, retrieval_service):
        """Test token estimation."""
        text = "This is a test text with multiple words."
        
        tokens = retrieval_service._estimate_tokens(text)
        
        assert tokens > 0
        assert tokens == len(text) // 4  # Should use the 4 chars per token estimation

    def test_estimate_tokens_empty(self, retrieval_service):
        """Test token estimation with empty text."""
        text = ""
        
        tokens = retrieval_service._estimate_tokens(text)
        
        assert tokens == 0

    def test_create_combined_context(self, retrieval_service, sample_search_results):
        """Test combined context creation."""
        context = retrieval_service._create_combined_context(sample_search_results)
        
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain text from all results
        for result in sample_search_results:
            assert result.text in context

    def test_create_combined_context_empty(self, retrieval_service):
        """Test combined context creation with empty results."""
        context = retrieval_service._create_combined_context([])
        
        assert context == ""

    def test_create_combined_context_error(self, retrieval_service, sample_search_results):
        """Test combined context creation error handling."""
        # Mock a result with None relevance_score to trigger error
        sample_search_results[0].relevance_score = None
        
        context = retrieval_service._create_combined_context(sample_search_results)
        
        # Should fallback to simple concatenation
        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_get_chunk_with_context_success(self, retrieval_service):
        """Test successful chunk retrieval with context."""
        mock_chunk_data = {
            "chunk_id": "chunk_1",
            "text": "Test chunk text",
            "document_id": "doc_1"
        }
        retrieval_service.es_service.get_chunk = AsyncMock(return_value=mock_chunk_data)
        
        result = await retrieval_service.get_chunk_with_context("chunk_1")
        
        assert result == mock_chunk_data
        retrieval_service.es_service.get_chunk.assert_called_once_with("chunk_1")

    @pytest.mark.asyncio
    async def test_get_chunk_with_context_error(self, retrieval_service):
        """Test chunk retrieval error handling."""
        retrieval_service.es_service.get_chunk = AsyncMock(side_effect=Exception("Chunk retrieval failed"))
        
        result = await retrieval_service.get_chunk_with_context("chunk_1")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_health_check_success(self, retrieval_service):
        """Test successful health check."""
        retrieval_service.es_service.client.cluster.health.return_value = {"status": "green"}
        retrieval_service.embedding_service.health_check = AsyncMock(return_value=True)
        
        result = await retrieval_service.health_check()
        
        assert result["elasticsearch"] is True
        assert result["embeddings"] is True

    @pytest.mark.asyncio
    async def test_health_check_elasticsearch_unhealthy(self, retrieval_service):
        """Test health check with unhealthy Elasticsearch."""
        retrieval_service.es_service.client.cluster.health.return_value = {"status": "red"}
        retrieval_service.embedding_service.health_check = AsyncMock(return_value=True)
        
        result = await retrieval_service.health_check()
        
        assert result["elasticsearch"] is False
        assert result["embeddings"] is True

    @pytest.mark.asyncio
    async def test_health_check_embeddings_unhealthy(self, retrieval_service):
        """Test health check with unhealthy embeddings."""
        retrieval_service.es_service.client.cluster.health.return_value = {"status": "green"}
        retrieval_service.embedding_service.health_check = AsyncMock(return_value=False)
        
        result = await retrieval_service.health_check()
        
        assert result["elasticsearch"] is True
        assert result["embeddings"] is False
