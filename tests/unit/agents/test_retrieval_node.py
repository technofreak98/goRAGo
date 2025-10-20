"""Unit tests for RetrievalNode."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.agents.retrieval_node import RetrievalNode
from app.models import SearchQuery, SearchResult, SearchResponse


@pytest.mark.unit
class TestRetrievalNode:
    """Test cases for RetrievalNode."""

    @pytest.fixture
    def retrieval_node(self):
        """Create RetrievalNode instance with mocked dependencies."""
        with patch('app.agents.retrieval_node.RetrievalService') as mock_retrieval, \
             patch('app.agents.retrieval_node.QueryPreprocessor') as mock_preprocessor, \
             patch('app.agents.retrieval_node.LocationExtractor') as mock_location, \
             patch('app.agents.retrieval_node.LLMService') as mock_llm:
            
            node = RetrievalNode()
            node.retrieval_service = mock_retrieval.return_value
            node.query_preprocessor = mock_preprocessor.return_value
            node.location_extractor = mock_location.return_value
            node.llm_service = mock_llm.return_value
            return node

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                chunk_id="chunk_1",
                text="Rome is the capital of Italy and has many historical sites.",
                score=0.95,
                document_id="doc_1",
                level=0,
                rank=1,
                relevance_score=0.95,
                token_count=15
            ),
            SearchResult(
                chunk_id="chunk_2", 
                text="Paris is the capital of France and is known for the Eiffel Tower.",
                score=0.88,
                document_id="doc_2",
                level=0,
                rank=2,
                relevance_score=0.88,
                token_count=16
            )
        ]

    @pytest.fixture
    def sample_search_response(self, sample_search_results):
        """Create sample search response."""
        return SearchResponse(
            query="Tell me about Rome",
            results=sample_search_results,
            total_results=2,
            processing_time_ms=150.0,
            used_reranking=True,
            used_compression=False,
            combined_context="Rome is the capital of Italy...",
            total_tokens=31,
            max_relevance_score=0.95,
            min_relevance_score=0.88
        )

    @pytest.mark.asyncio
    async def test_retrieve_documents_weather_combined_route_extracts_places(self, retrieval_node, sample_search_response):
        """Test that places are extracted for combined route (weather + document)."""
        # Setup mocks
        retrieval_node.query_preprocessor.preprocess_query.return_value = ("processed query", ["keywords"])
        retrieval_node.retrieval_service.search = AsyncMock(return_value=sample_search_response)
        retrieval_node.llm_service.extract_places_from_text = AsyncMock(return_value=["Rome", "Italy"])
        
        state = {
            "query": "What's the weather like in Rome mentioned in the documents?",
            "route": "combined"
        }
        
        result = await retrieval_node.retrieve_documents(state)
        
        # Verify places were extracted
        assert set(result["extracted_places"]) == {"Rome", "Italy"}
        retrieval_node.llm_service.extract_places_from_text.assert_called_once()
        
        # Verify other state updates
        assert result["document_context"] is not None
        assert len(result["document_sources"]) == 2

    @pytest.mark.asyncio
    async def test_retrieve_documents_weather_only_route_extracts_places(self, retrieval_node, sample_search_response):
        """Test that places are extracted for weather_only route."""
        # Setup mocks
        retrieval_node.query_preprocessor.preprocess_query.return_value = ("processed query", ["keywords"])
        retrieval_node.retrieval_service.search = AsyncMock(return_value=sample_search_response)
        retrieval_node.llm_service.extract_places_from_text = AsyncMock(return_value=["Paris", "France"])
        
        state = {
            "query": "What's the weather in Paris?",
            "route": "weather_only"
        }
        
        result = await retrieval_node.retrieve_documents(state)
        
        # Verify places were extracted
        assert set(result["extracted_places"]) == {"Paris", "France"}
        retrieval_node.llm_service.extract_places_from_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_documents_document_only_route_no_place_extraction(self, retrieval_node, sample_search_response):
        """Test that places are NOT extracted for document_only route."""
        # Setup mocks
        retrieval_node.query_preprocessor.preprocess_query.return_value = ("processed query", ["keywords"])
        retrieval_node.retrieval_service.search = AsyncMock(return_value=sample_search_response)
        retrieval_node.llm_service.extract_places_from_text = AsyncMock(return_value=["Rome", "Italy"])
        
        state = {
            "query": "Tell me about the history of Rome",
            "route": "document_only"
        }
        
        result = await retrieval_node.retrieve_documents(state)
        
        # Verify places were NOT extracted
        assert result["extracted_places"] == []
        retrieval_node.llm_service.extract_places_from_text.assert_not_called()
        
        # Verify other state updates still work
        assert result["document_context"] is not None
        assert len(result["document_sources"]) == 2

    @pytest.mark.asyncio
    async def test_retrieve_documents_out_of_scope_route_no_place_extraction(self, retrieval_node, sample_search_response):
        """Test that places are NOT extracted for out_of_scope route."""
        # Setup mocks
        retrieval_node.query_preprocessor.preprocess_query.return_value = ("processed query", ["keywords"])
        retrieval_node.retrieval_service.search = AsyncMock(return_value=sample_search_response)
        retrieval_node.llm_service.extract_places_from_text = AsyncMock(return_value=["Rome", "Italy"])
        
        state = {
            "query": "How to cook pasta",
            "route": "out_of_scope"
        }
        
        result = await retrieval_node.retrieve_documents(state)
        
        # Verify places were NOT extracted
        assert result["extracted_places"] == []
        retrieval_node.llm_service.extract_places_from_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_documents_empty_query(self, retrieval_node):
        """Test handling of empty query."""
        state = {"query": ""}
        
        result = await retrieval_node.retrieve_documents(state)
        
        assert result["document_context"] == ""
        assert result["document_sources"] == []
        assert result["extracted_places"] == []

    @pytest.mark.asyncio
    async def test_retrieve_documents_no_results(self, retrieval_node):
        """Test handling when no documents are found."""
        # Setup mocks
        retrieval_node.query_preprocessor.preprocess_query.return_value = ("processed query", ["keywords"])
        empty_response = SearchResponse(
            query="processed query",
            results=[],
            total_results=0,
            processing_time_ms=100.0,
            used_reranking=False,
            used_compression=False,
            combined_context="",
            total_tokens=0,
            max_relevance_score=0.0,
            min_relevance_score=0.0
        )
        retrieval_node.retrieval_service.search = AsyncMock(return_value=empty_response)
        
        state = {
            "query": "Tell me about unicorns",
            "route": "document_only"
        }
        
        result = await retrieval_node.retrieve_documents(state)
        
        assert result["document_context"] == "No relevant documents found."
        assert result["document_sources"] == []
        assert result["extracted_places"] == []

    @pytest.mark.asyncio
    async def test_retrieve_documents_error_handling(self, retrieval_node):
        """Test error handling in retrieve_documents."""
        # Setup mocks to raise an error
        retrieval_node.query_preprocessor.preprocess_query.side_effect = Exception("Preprocessing failed")
        
        state = {
            "query": "Tell me about Rome",
            "route": "document_only"
        }
        
        result = await retrieval_node.retrieve_documents(state)
        
        assert "Document retrieval error" in result["document_context"]
        assert result["document_sources"] == []
        assert result["extracted_places"] == []
