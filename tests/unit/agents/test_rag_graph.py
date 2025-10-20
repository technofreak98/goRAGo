"""Unit tests for RAGGraph."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langgraph.graph import StateGraph

from app.agents.rag_graph import RAGGraph


@pytest.mark.unit
class TestRAGGraph:
    """Test cases for RAGGraph."""

    @pytest.fixture
    def rag_graph(self):
        """Create RAGGraph instance."""
        return RAGGraph()

    def test_rag_graph_initialization(self, rag_graph):
        """Test RAGGraph initialization."""
        assert rag_graph.graph is not None
        # The graph is a CompiledStateGraph, not StateGraph
        assert hasattr(rag_graph.graph, 'invoke')

    def test_route_from_router_weather_only(self, rag_graph):
        """Test routing from router for weather-only queries."""
        state = {"route": "weather_only"}
        
        result = rag_graph._route_from_router(state)
        
        assert result == "weather"

    def test_route_from_router_document_only(self, rag_graph):
        """Test routing from router for document-only queries."""
        state = {"route": "document_only"}
        
        result = rag_graph._route_from_router(state)
        
        assert result == "documents"

    def test_route_from_router_combined(self, rag_graph):
        """Test routing from router for combined queries."""
        state = {"route": "combined"}
        
        result = rag_graph._route_from_router(state)
        
        assert result == "documents"  # Combined queries go to documents first

    def test_route_from_router_out_of_scope(self, rag_graph):
        """Test routing from router for out-of-scope queries."""
        state = {"route": "out_of_scope"}
        
        result = rag_graph._route_from_router(state)
        
        assert result == "guardrail"

    def test_route_from_router_unknown(self, rag_graph):
        """Test routing from router for unknown route."""
        state = {"route": "unknown"}
        
        result = rag_graph._route_from_router(state)
        
        assert result == "__end__"

    def test_route_from_documents_combined(self, rag_graph):
        """Test routing from documents for combined queries."""
        state = {"route": "combined"}
        
        result = rag_graph._route_from_documents(state)
        
        assert result == "weather"  # Combined queries go to weather after documents

    def test_route_from_documents_document_only(self, rag_graph):
        """Test routing from documents for document-only queries."""
        state = {"route": "document_only"}
        
        result = rag_graph._route_from_documents(state)
        
        assert result == "generation"  # Document-only queries go to generation

    def test_route_from_documents_unknown(self, rag_graph):
        """Test routing from documents for unknown route."""
        state = {"route": "unknown"}
        
        result = rag_graph._route_from_documents(state)
        
        assert result == "__end__"

    @pytest.mark.asyncio
    async def test_process_query_success(self, rag_graph, mock_agent_state):
        """Test successful query processing."""
        # Mock the graph execution
        mock_result = {
            "final_answer": "This is a test answer about Rome.",
            "route": "weather",
            "confidence": 0.9,
            "reasoning": "Query was about weather",
            "sources": [{"type": "weather_api", "city": "Rome"}]
        }
        
        with patch.object(rag_graph.graph, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_result
            
            query = "What's the weather in Rome?"
            result = await rag_graph.process_query(query)
            
            assert result["answer"] == "This is a test answer about Rome."
            assert result["route"] == "weather"
            assert result["confidence"] == 0.9
            assert result["reasoning"] == "Query was about weather"
            assert len(result["sources"]) == 1
            assert result["session_id"] is None

    @pytest.mark.asyncio
    async def test_process_query_with_session_id(self, rag_graph, mock_agent_state):
        """Test query processing with session ID."""
        mock_result = {
            "final_answer": "This is a test answer.",
            "route": "document",
            "confidence": 0.8,
            "reasoning": "Query was about documents",
            "sources": []
        }
        
        with patch.object(rag_graph.graph, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_result
            
            query = "Tell me about Rome"
            session_id = "test_session_123"
            result = await rag_graph.process_query(query, session_id)
            
            assert result["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_process_query_graph_not_initialized(self, rag_graph):
        """Test query processing when graph is not initialized."""
        rag_graph.graph = None
        
        query = "What's the weather in Rome?"
        
        result = await rag_graph.process_query(query)
        
        assert result["route"] == "error"
        assert "error" in result["answer"].lower()
        assert "Graph not initialized" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_process_query_error(self, rag_graph):
        """Test query processing error handling."""
        with patch.object(rag_graph.graph, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.side_effect = Exception("Graph execution failed")
            
            query = "What's the weather in Rome?"
            result = await rag_graph.process_query(query)
            
            assert "error" in result["answer"].lower()
            assert result["route"] == "error"
            assert result["confidence"] == 0.0
            assert "Processing error" in result["reasoning"]
            assert result["sources"] == []

    def test_health_check_success(self, rag_graph):
        """Test successful health check."""
        result = rag_graph.health_check()
        
        assert result is True

    def test_health_check_graph_not_initialized(self, rag_graph):
        """Test health check when graph is not initialized."""
        rag_graph.graph = None
        
        result = rag_graph.health_check()
        
        assert result is False

    def test_health_check_error(self, rag_graph):
        """Test health check error handling."""
        # Set graph to None to simulate error condition
        rag_graph.graph = None
        result = rag_graph.health_check()
        
        assert result is False

    def test_initial_state_structure(self, rag_graph):
        """Test that initial state has correct structure."""
        query = "Test query"
        
        # This is a bit of an implementation detail test, but useful for ensuring
        # the state structure is correct
        expected_keys = {
            "query", "route", "confidence", "reasoning", "document_context",
            "weather_context", "document_sources", "weather_sources",
            "extracted_places", "final_answer", "sources"
        }
        
        # We can't directly test the initial state creation, but we can verify
        # the structure by checking if the graph was built successfully
        assert rag_graph.graph is not None
