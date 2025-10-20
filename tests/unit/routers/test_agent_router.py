"""Unit tests for agent router."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException

from app.routers.agent import router, get_agent, process_agent_query, agent_health_check, get_agent_info
from app.models import AgentQuery, AgentResponse


@pytest.mark.unit
class TestAgentRouter:
    """Test cases for agent router."""

    def test_get_agent_singleton(self):
        """Test that get_agent returns singleton instance."""
        # Clear the global instance
        import app.routers.agent
        app.routers.agent._agent_instance = None
        
        agent1 = get_agent()
        agent2 = get_agent()
        
        assert agent1 is agent2
        assert isinstance(agent1, type(agent1))

    @pytest.mark.asyncio
    async def test_process_agent_query_success(self):
        """Test successful agent query processing."""
        mock_agent = Mock()
        mock_agent.process_query = AsyncMock(return_value=AgentResponse(
            answer="This is a test answer about Rome.",
            route_taken="weather",
            sources=[],
            weather_data=[],
            locations=[],
            processing_time_ms=150.0,
            error=None
        ))
        
        with patch('app.routers.agent.get_agent', return_value=mock_agent):
            query = AgentQuery(query="What's the weather in Rome?")
            result = await process_agent_query(query)
            
            assert isinstance(result, AgentResponse)
            assert result.answer == "This is a test answer about Rome."
            assert result.route_taken == "weather"
            assert result.processing_time_ms == 150.0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_process_agent_query_empty_query(self):
        """Test agent query processing with empty query."""
        query = AgentQuery(query="   ")  # Empty/whitespace query
        
        with pytest.raises(HTTPException) as exc_info:
            await process_agent_query(query)
        
        assert exc_info.value.status_code == 400
        assert "Query cannot be empty" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_agent_query_agent_error(self):
        """Test agent query processing when agent raises error."""
        mock_agent = Mock()
        mock_agent.process_query = AsyncMock(side_effect=Exception("Agent processing failed"))
        
        with patch('app.routers.agent.get_agent', return_value=mock_agent):
            query = AgentQuery(query="What's the weather in Rome?")
            
            with pytest.raises(HTTPException) as exc_info:
                await process_agent_query(query)
            
            assert exc_info.value.status_code == 500
            assert "Agent processing failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_agent_health_check_success(self):
        """Test successful agent health check."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value={
            "rag_graph": True,
            "modular_agents": True
        })
        
        with patch('app.routers.agent.get_agent', return_value=mock_agent):
            result = await agent_health_check()
            
            assert result["status"] == "healthy"
            assert result["components"]["rag_graph"] is True
            assert result["components"]["modular_agents"] is True
            assert result["agent_type"] == "LangGraph Router Agent"

    @pytest.mark.asyncio
    async def test_agent_health_check_unhealthy(self):
        """Test agent health check with unhealthy components."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value={
            "rag_graph": False,
            "modular_agents": True
        })
        
        with patch('app.routers.agent.get_agent', return_value=mock_agent):
            result = await agent_health_check()
            
            assert result["status"] == "unhealthy"
            assert result["components"]["rag_graph"] is False
            assert result["components"]["modular_agents"] is True

    @pytest.mark.asyncio
    async def test_agent_health_check_error(self):
        """Test agent health check error handling."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        
        with patch('app.routers.agent.get_agent', return_value=mock_agent):
            result = await agent_health_check()
            
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert result["agent_type"] == "LangGraph Router Agent"

    @pytest.mark.asyncio
    async def test_get_agent_info(self):
        """Test agent info retrieval."""
        result = await get_agent_info()
        
        assert result["agent_type"] == "LangGraph Router Agent"
        assert "capabilities" in result
        assert "supported_routes" in result
        assert "example_queries" in result
        
        # Check specific capabilities
        capabilities = result["capabilities"]
        assert "Document retrieval and literature questions" in capabilities
        assert "Weather information for travel planning" in capabilities
        assert "Combined literature + weather queries" in capabilities
        assert "Intelligent query routing" in capabilities
        
        # Check supported routes
        routes = result["supported_routes"]
        assert any("document" in route for route in routes)
        assert any("weather" in route for route in routes)
        assert any("combined" in route for route in routes)
        assert any("guardrails" in route for route in routes)
        
        # Check example queries
        examples = result["example_queries"]
        assert len(examples) > 0
        assert any("Mark Twain" in query for query in examples)
        assert any("weather" in query.lower() for query in examples)

    def test_router_initialization(self):
        """Test that router is properly initialized."""
        assert router.prefix == "/api/agent"
        assert "agent" in router.tags
        
        # Check that routes are registered
        route_paths = [route.path for route in router.routes]
        assert "/api/agent/query" in route_paths
        assert "/api/agent/health" in route_paths
        assert "/api/agent/info" in route_paths

    @pytest.mark.asyncio
    async def test_process_agent_query_http_exception_passthrough(self):
        """Test that HTTPException is passed through without modification."""
        query = AgentQuery(query="   ")  # Empty query
        
        with pytest.raises(HTTPException) as exc_info:
            await process_agent_query(query)
        
        # Should be the original HTTPException, not wrapped
        assert exc_info.value.status_code == 400
        assert "Query cannot be empty" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_agent_query_general_exception_wrapping(self):
        """Test that general exceptions are wrapped in HTTPException."""
        mock_agent = Mock()
        mock_agent.process_query = AsyncMock(side_effect=ValueError("Invalid input"))
        
        with patch('app.routers.agent.get_agent', return_value=mock_agent):
            query = AgentQuery(query="What's the weather in Rome?")
            
            with pytest.raises(HTTPException) as exc_info:
                await process_agent_query(query)
            
            assert exc_info.value.status_code == 500
            assert "Agent processing failed" in exc_info.value.detail
            assert "Invalid input" in exc_info.value.detail
