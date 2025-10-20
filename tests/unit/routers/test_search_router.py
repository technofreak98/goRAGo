"""Unit tests for search router."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException

from app.routers.search import router, search_documents, get_chunk, search_health_check
from app.models import SearchQuery, SearchResponse, SearchResult, ChunkResponse


@pytest.mark.unit
class TestSearchRouter:
    """Test cases for search router."""

    @pytest.mark.asyncio
    async def test_search_documents_success(self):
        """Test successful document search."""
        mock_search_response = SearchResponse(
            query="Rome Italy travel",
            results=[
                SearchResult(
                    chunk_id="chunk_1",
                    text="This is about Rome and Italy travel.",
                    score=0.95,
                    document_id="doc_1",
                    level=0,
                    rank=1,
                    relevance_score=0.95,
                    token_count=10
                )
            ],
            total_results=1,
            processing_time_ms=150.0,
            used_reranking=True,
            used_compression=False,
            combined_context="Test context",
            total_tokens=10,
            max_relevance_score=0.95,
            min_relevance_score=0.95
        )
        
        mock_retrieval_service = Mock()
        mock_retrieval_service.search = AsyncMock(return_value=mock_search_response)
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            from app.routers.search import SimpleSearchQuery
            
            search_query = SimpleSearchQuery(
                query="Rome Italy travel",
                top_k=10,
                rerank=True,
                compression=False,
                filter_by_part=1,
                filter_by_chapter=2
            )
            
            result = await search_documents(search_query)
            
            assert isinstance(result, SearchResponse)
            assert result.query == "Rome Italy travel"
            assert len(result.results) == 1
            assert result.results[0].chunk_id == "chunk_1"
            assert result.used_reranking is True
            assert result.used_compression is False
            
            # Verify that RetrievalService was called with correct SearchQuery
            mock_retrieval_service.search.assert_called_once()
            call_args = mock_retrieval_service.search.call_args[0][0]
            assert isinstance(call_args, SearchQuery)
            assert call_args.query == "Rome Italy travel"
            assert call_args.top_k == 10
            assert call_args.rerank is True
            assert call_args.compression is False
            assert call_args.filter_by_part == 1
            assert call_args.filter_by_chapter == 2

    @pytest.mark.asyncio
    async def test_search_documents_error(self):
        """Test document search error handling."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.search = AsyncMock(side_effect=Exception("Search failed"))
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            from app.routers.search import SimpleSearchQuery
            
            search_query = SimpleSearchQuery(query="Rome Italy travel")
            
            with pytest.raises(HTTPException) as exc_info:
                await search_documents(search_query)
            
            assert exc_info.value.status_code == 500
            assert "Search failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_chunk_success(self):
        """Test successful chunk retrieval."""
        mock_chunk_data = {
            "chunk_id": "chunk_1",
            "text": "This is a test chunk about Rome.",
            "document_id": "doc_1",
            "level": 0,
            "parent_id": None,
            "child_ids": [],
            "section_info": {"section_number": 1, "title": "Introduction"},
            "chapter_info": {"chapter_number": 1, "title": "Getting Started"},
            "part_info": {"part_number": 1, "title": "Part I"},
            "parent_window": "Context about Italy"
        }
        
        mock_retrieval_service = Mock()
        mock_retrieval_service.get_chunk_with_context = AsyncMock(return_value=mock_chunk_data)
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            result = await get_chunk("chunk_1")
            
            assert isinstance(result, ChunkResponse)
            assert result.chunk_id == "chunk_1"
            assert result.text == "This is a test chunk about Rome."
            assert result.document_id == "doc_1"
            assert result.level == 0
            assert result.parent_id is None
            assert result.child_ids == []
            
            mock_retrieval_service.get_chunk_with_context.assert_called_once_with("chunk_1")

    @pytest.mark.asyncio
    async def test_get_chunk_not_found(self):
        """Test chunk retrieval when chunk not found."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.get_chunk_with_context = AsyncMock(return_value=None)
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            with pytest.raises(HTTPException) as exc_info:
                await get_chunk("nonexistent_chunk")
            
            assert exc_info.value.status_code == 404
            assert "Chunk not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_chunk_error(self):
        """Test chunk retrieval error handling."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.get_chunk_with_context = AsyncMock(side_effect=Exception("Retrieval failed"))
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            with pytest.raises(HTTPException) as exc_info:
                await get_chunk("chunk_1")
            
            assert exc_info.value.status_code == 500
            assert "Failed to retrieve chunk" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_search_health_check_success(self):
        """Test successful search health check."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.health_check = AsyncMock(return_value={
            "elasticsearch": True,
            "embeddings": True
        })
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            result = await search_health_check()
            
            assert result["status"] == "healthy"
            assert result["components"]["elasticsearch"] is True
            assert result["components"]["embeddings"] is True

    @pytest.mark.asyncio
    async def test_search_health_check_unhealthy(self):
        """Test search health check with unhealthy components."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.health_check = AsyncMock(return_value={
            "elasticsearch": False,
            "embeddings": True
        })
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            result = await search_health_check()
            
            assert result["status"] == "unhealthy"
            assert result["components"]["elasticsearch"] is False
            assert result["components"]["embeddings"] is True

    @pytest.mark.asyncio
    async def test_search_health_check_error(self):
        """Test search health check error handling."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            result = await search_health_check()
            
            assert result["status"] == "unhealthy"
            assert "error" in result

    def test_router_initialization(self):
        """Test that router is properly initialized."""
        assert router.prefix == "/api/search"
        assert "search" in router.tags
        
        # Check that routes are registered
        route_paths = [route.path for route in router.routes]
        assert "/api/search/search" in route_paths
        assert "/api/search/chunk/{chunk_id}" in route_paths
        assert "/api/search/health" in route_paths

    def test_simple_search_query_model(self):
        """Test SimpleSearchQuery model validation."""
        from app.routers.search import SimpleSearchQuery
        
        # Test with all fields
        query = SimpleSearchQuery(
            query="Rome Italy travel",
            top_k=20,
            rerank=False,
            compression=True,
            filter_by_part=1,
            filter_by_chapter=2
        )
        
        assert query.query == "Rome Italy travel"
        assert query.top_k == 20
        assert query.rerank is False
        assert query.compression is True
        assert query.filter_by_part == 1
        assert query.filter_by_chapter == 2

    def test_simple_search_query_defaults(self):
        """Test SimpleSearchQuery default values."""
        from app.routers.search import SimpleSearchQuery
        
        # Test with minimal fields
        query = SimpleSearchQuery(query="Rome Italy travel")
        
        assert query.query == "Rome Italy travel"
        assert query.top_k == 10  # Default value
        assert query.rerank is True  # Default value
        assert query.compression is True  # Default value
        assert query.filter_by_part is None  # Default value
        assert query.filter_by_chapter is None  # Default value

    @pytest.mark.asyncio
    async def test_get_chunk_http_exception_passthrough(self):
        """Test that HTTPException is passed through without modification."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.get_chunk_with_context = AsyncMock(return_value=None)
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            with pytest.raises(HTTPException) as exc_info:
                await get_chunk("nonexistent_chunk")
            
            # Should be the original HTTPException, not wrapped
            assert exc_info.value.status_code == 404
            assert "Chunk not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_chunk_general_exception_wrapping(self):
        """Test that general exceptions are wrapped in HTTPException."""
        mock_retrieval_service = Mock()
        mock_retrieval_service.get_chunk_with_context = AsyncMock(side_effect=ValueError("Invalid chunk ID"))
        
        with patch('app.routers.search.RetrievalService', return_value=mock_retrieval_service):
            with pytest.raises(HTTPException) as exc_info:
                await get_chunk("chunk_1")
            
            assert exc_info.value.status_code == 500
            assert "Failed to retrieve chunk" in exc_info.value.detail
            assert "Invalid chunk ID" in exc_info.value.detail
