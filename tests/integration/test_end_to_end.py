"""End-to-end integration tests for the RAG-ES system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from app.services.langgraph_agent import LangGraphAgent
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.elasticsearch_service import ElasticsearchService
from app.models import SearchQuery, AgentQuery


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def mock_services(self):
        """Create mocked services for integration testing."""
        # Mock Elasticsearch service
        mock_es_service = Mock()
        mock_es_service.hybrid_search = AsyncMock(return_value=[])
        mock_es_service.get_chunk = AsyncMock(return_value=None)
        mock_es_service.client.cluster.health = Mock(return_value={"status": "green"})
        
        # Mock Embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.generate_query_embedding = AsyncMock(return_value=[0.1] * 1536)
        mock_embedding_service.health_check = AsyncMock(return_value=True)
        
        # Mock LLM service
        mock_llm_service = Mock()
        mock_llm_service.generate_answer = AsyncMock(return_value="This is a test answer about Rome.")
        mock_llm_service.classify_query_intent = AsyncMock(return_value="document")
        mock_llm_service.health_check = AsyncMock(return_value=True)
        
        return {
            "es_service": mock_es_service,
            "embedding_service": mock_embedding_service,
            "llm_service": mock_llm_service
        }

    @pytest.mark.asyncio
    async def test_document_search_workflow(self, mock_services):
        """Test complete document search workflow."""
        # Mock search results
        from app.models import SearchResult
        mock_results = [
            SearchResult(
                chunk_id="chunk_1",
                text="Rome is the capital of Italy and has many historical sites.",
                score=0.95,
                document_id="doc_1",
                level=0,
                rank=1,
                relevance_score=0.95,
                token_count=15
            )
        ]
        mock_services["es_service"].hybrid_search.return_value = mock_results
        
        # Create retrieval service with mocked dependencies
        with patch('app.services.retrieval_service.ElasticsearchService', return_value=mock_services["es_service"]), \
             patch('app.services.retrieval_service.EmbeddingService', return_value=mock_services["embedding_service"]):
            
            retrieval_service = RetrievalService()
            
            # Test search
            query = SearchQuery(
                query="Tell me about Rome",
                top_k=10,
                rerank=True,
                compression=True
            )
            
            result = await retrieval_service.search(query)
            
            assert result.query == "Tell me about Rome"
            assert len(result.results) == 1
            assert result.results[0].chunk_id == "chunk_1"
            assert "Rome" in result.results[0].text
            assert result.used_reranking is True
            assert result.used_compression is True

    @pytest.mark.asyncio
    async def test_agent_query_workflow(self, mock_services):
        """Test complete agent query workflow."""
        # Mock RAG graph result
        mock_rag_result = {
            "final_answer": "Rome is the capital of Italy and has many historical sites.",
            "route": "document",
            "sources": [
                {
                    "type": "document",
                    "chunk_id": "chunk_1",
                    "text": "Rome is the capital of Italy.",
                    "score": 0.95
                }
            ]
        }
        
        # Mock RAG graph
        mock_rag_graph = Mock()
        mock_rag_graph.process_query = AsyncMock(return_value=mock_rag_result)
        mock_rag_graph.health_check = Mock(return_value=True)
        
        # Create agent with mocked RAG graph
        with patch('app.services.langgraph_agent.rag_graph', mock_rag_graph):
            agent = LangGraphAgent()
            
            # Test query processing
            query = "Tell me about Rome"
            result = await agent.process_query(query)
            
            assert result.answer == "Rome is the capital of Italy and has many historical sites."
            assert result.route_taken == "document"
            assert len(result.sources) == 1
            assert result.sources[0]["type"] == "document"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_weather_query_workflow(self, mock_services):
        """Test weather query workflow."""
        # Mock weather data
        mock_weather_data = [
            {
                "type": "weather_api",
                "city": "Rome",
                "country": "IT",
                "temperature": {"current": 22.5, "min": 18.0, "max": 26.0},
                "conditions": {"description": "Partly cloudy", "main": "Clouds"},
                "humidity": 65,
                "pressure": 1013,
                "wind": {"speed": 3.2, "direction": 180},
                "visibility": 10.0,
                "cloudiness": 40,
                "timestamp": 1640995200
            }
        ]
        
        # Mock RAG graph result with weather
        mock_rag_result = {
            "final_answer": "Rome currently has partly cloudy weather with a temperature of 22.5°C.",
            "route": "weather",
            "sources": mock_weather_data
        }
        
        # Mock RAG graph
        mock_rag_graph = Mock()
        mock_rag_graph.process_query = AsyncMock(return_value=mock_rag_result)
        mock_rag_graph.health_check = Mock(return_value=True)
        
        # Create agent with mocked RAG graph
        with patch('app.services.langgraph_agent.rag_graph', mock_rag_graph):
            agent = LangGraphAgent()
            
            # Test weather query
            query = "What's the weather in Rome?"
            result = await agent.process_query(query)
            
            assert "weather" in result.answer.lower()
            assert "22.5" in result.answer
            assert result.route_taken == "weather"
            assert len(result.weather_data) == 1
            assert result.weather_data[0].city == "Rome"
            assert len(result.locations) == 1
            assert result.locations[0].name == "Rome"

    @pytest.mark.asyncio
    async def test_combined_query_workflow(self, mock_services):
        """Test combined literature + weather query workflow."""
        # Mock combined result
        mock_combined_result = {
            "final_answer": "Rome is mentioned in many literary works and currently has partly cloudy weather.",
            "route": "combined",
            "sources": [
                {
                    "type": "document",
                    "chunk_id": "chunk_1",
                    "text": "Rome is mentioned in literature.",
                    "score": 0.95
                },
                {
                    "type": "weather_api",
                    "city": "Rome",
                    "country": "IT",
                    "temperature": {"current": 22.5},
                    "conditions": {"description": "Partly cloudy"},
                    "humidity": 65,
                    "pressure": 1013,
                    "wind": {"speed": 3.2},
                    "visibility": 10.0,
                    "cloudiness": 40,
                    "timestamp": 1640995200
                }
            ]
        }
        
        # Mock RAG graph
        mock_rag_graph = Mock()
        mock_rag_graph.process_query = AsyncMock(return_value=mock_combined_result)
        mock_rag_graph.health_check = Mock(return_value=True)
        
        # Create agent with mocked RAG graph
        with patch('app.services.langgraph_agent.rag_graph', mock_rag_graph):
            agent = LangGraphAgent()
            
            # Test combined query
            query = "I want to visit places mentioned in literature in Rome - what's the weather?"
            result = await agent.process_query(query)
            
            assert "literature" in result.answer.lower()
            assert "weather" in result.answer.lower()
            assert result.route_taken == "combined"
            assert len(result.sources) == 2
            assert len(result.weather_data) == 1
            assert len(result.locations) == 1

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_services):
        """Test error handling in the workflow."""
        # Mock RAG graph that raises an error
        mock_rag_graph = Mock()
        mock_rag_graph.process_query = AsyncMock(side_effect=Exception("Processing failed"))
        mock_rag_graph.health_check = Mock(return_value=True)
        
        # Create agent with mocked RAG graph
        with patch('app.services.langgraph_agent.rag_graph', mock_rag_graph):
            agent = LangGraphAgent()
            
            # Test error handling
            query = "Tell me about Rome"
            result = await agent.process_query(query)
            
            assert "error" in result.answer.lower()
            assert result.route_taken == "error"
            assert result.error == "Processing failed"
            assert result.sources == []
            assert result.weather_data == []
            assert result.locations == []

    @pytest.mark.asyncio
    async def test_health_check_workflow(self, mock_services):
        """Test health check workflow across services."""
        # Mock all services as healthy
        mock_services["es_service"].client.cluster.health.return_value = {"status": "green"}
        mock_services["embedding_service"].health_check.return_value = True
        mock_services["llm_service"].health_check.return_value = True
        
        # Test retrieval service health check
        with patch('app.services.retrieval_service.ElasticsearchService', return_value=mock_services["es_service"]), \
             patch('app.services.retrieval_service.EmbeddingService', return_value=mock_services["embedding_service"]):
            
            retrieval_service = RetrievalService()
            health_status = await retrieval_service.health_check()
            
            assert health_status["elasticsearch"] is True
            assert health_status["embeddings"] is True

    @pytest.mark.asyncio
    async def test_search_with_filters_workflow(self, mock_services):
        """Test search workflow with filters."""
        # Mock search results
        from app.models import SearchResult
        mock_results = [
            SearchResult(
                chunk_id="chunk_1",
                text="This is about Rome in Part I, Chapter 2.",
                score=0.95,
                document_id="doc_1",
                level=0,
                rank=1,
                relevance_score=0.95,
                token_count=10
            )
        ]
        mock_services["es_service"].hybrid_search.return_value = mock_results
        
        # Create retrieval service with mocked dependencies
        with patch('app.services.retrieval_service.ElasticsearchService', return_value=mock_services["es_service"]), \
             patch('app.services.retrieval_service.EmbeddingService', return_value=mock_services["embedding_service"]):
            
            retrieval_service = RetrievalService()
            
            # Test search with filters
            query = SearchQuery(
                query="Tell me about Rome",
                top_k=10,
                rerank=True,
                compression=True,
                filter_by_part=1,
                filter_by_chapter=2
            )
            
            result = await retrieval_service.search(query)
            
            assert result.query == "Tell me about Rome"
            assert len(result.results) == 1
            # Verify filters were passed to Elasticsearch
            mock_services["es_service"].hybrid_search.assert_called_once()
            call_args = mock_services["es_service"].hybrid_search.call_args
            filters = call_args[1]["filters"]
            assert filters["part_number"] == 1
            assert filters["chapter_number"] == 2

    @pytest.mark.asyncio
    async def test_reranking_workflow(self, mock_services):
        """Test reranking workflow."""
        # Mock initial search results (more than final_top_k)
        from app.models import SearchResult
        mock_initial_results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                text=f"Text about Rome {i}",
                score=0.9 - (i * 0.1),
                document_id="doc_1",
                level=0,
                rank=i + 1,
                relevance_score=0.9 - (i * 0.1),
                token_count=10
            )
            for i in range(15)  # More than final_top_k (10)
        ]
        mock_services["es_service"].hybrid_search.return_value = mock_initial_results
        
        # Create retrieval service with mocked dependencies
        with patch('app.services.retrieval_service.ElasticsearchService', return_value=mock_services["es_service"]), \
             patch('app.services.retrieval_service.EmbeddingService', return_value=mock_services["embedding_service"]):
            
            retrieval_service = RetrievalService()
            
            # Test search with reranking
            query = SearchQuery(
                query="Tell me about Rome",
                top_k=10,
                rerank=True,
                compression=False
            )
            
            result = await retrieval_service.search(query)
            
            assert result.query == "Tell me about Rome"
            assert result.used_reranking is True
            # Results should be reranked and limited to final_top_k
            assert len(result.results) <= 10

    @pytest.mark.asyncio
    async def test_compression_workflow(self, mock_services):
        """Test compression workflow."""
        # Mock search results
        from app.models import SearchResult
        mock_results = [
            SearchResult(
                chunk_id="chunk_1",
                text="This is a very long text about Rome and Italy that contains many details about the city, its history, culture, and attractions. " * 10,  # Long text
                score=0.95,
                document_id="doc_1",
                level=0,
                rank=1,
                relevance_score=0.95,
                token_count=1000  # High token count
            )
        ]
        mock_services["es_service"].hybrid_search.return_value = mock_results
        
        # Create retrieval service with mocked dependencies
        with patch('app.services.retrieval_service.ElasticsearchService', return_value=mock_services["es_service"]), \
             patch('app.services.retrieval_service.EmbeddingService', return_value=mock_services["embedding_service"]):
            
            retrieval_service = RetrievalService()
            
            # Test search with compression
            query = SearchQuery(
                query="Tell me about Rome",
                top_k=10,
                rerank=False,
                compression=True
            )
            
            result = await retrieval_service.search(query)
            
            assert result.query == "Tell me about Rome"
            assert result.used_compression is True
            # Text should be compressed (shorter)
            if result.results:
                assert len(result.results[0].text) < len(mock_results[0].text)
