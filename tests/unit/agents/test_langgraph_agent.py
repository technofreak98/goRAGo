"""Unit tests for LangGraphAgent."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.services.langgraph_agent import LangGraphAgent
from app.models import AgentResponse, WeatherData, LocationInfo


@pytest.mark.unit
class TestLangGraphAgent:
    """Test cases for LangGraphAgent."""

    @pytest.fixture
    def langgraph_agent(self):
        """Create LangGraphAgent instance with mocked RAG graph."""
        with patch('app.services.langgraph_agent.rag_graph') as mock_rag_graph:
            agent = LangGraphAgent()
            agent.rag_graph = mock_rag_graph
            return agent

    @pytest.mark.asyncio
    async def test_process_query_success(self, langgraph_agent):
        """Test successful query processing."""
        mock_result = {
            "final_answer": "This is a test answer about Rome.",
            "route": "weather",
            "sources": [
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
        
        langgraph_agent.rag_graph.process_query = AsyncMock(return_value=mock_result)
        
        query = "What's the weather in Rome?"
        result = await langgraph_agent.process_query(query)
        
        assert isinstance(result, AgentResponse)
        assert result.answer == "This is a test answer about Rome."
        assert result.route_taken == "weather"
        assert len(result.sources) == 1
        assert len(result.weather_data) == 1
        assert len(result.locations) == 1
        assert result.weather_data[0].city == "Rome"
        assert result.locations[0].name == "Rome"
        assert result.error is None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_query_document_route(self, langgraph_agent):
        """Test query processing with document route."""
        mock_result = {
            "answer": "Rome is the capital of Italy and has many historical sites.",
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
        
        langgraph_agent.rag_graph.process_query = AsyncMock(return_value=mock_result)
        
        query = "Tell me about Rome"
        result = await langgraph_agent.process_query(query)
        
        assert result.answer == "Rome is the capital of Italy and has many historical sites."
        assert result.route_taken == "document"
        assert len(result.sources) == 1
        assert result.sources[0]["type"] == "document"
        assert len(result.weather_data) == 0
        assert len(result.locations) == 0

    @pytest.mark.asyncio
    async def test_process_query_combined_route(self, langgraph_agent):
        """Test query processing with combined route."""
        mock_result = {
            "final_answer": "Rome is beautiful and currently has partly cloudy weather.",
            "route": "combined",
            "sources": [
                {
                    "type": "document",
                    "chunk_id": "chunk_1",
                    "text": "Rome is the capital of Italy.",
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
        
        langgraph_agent.rag_graph.process_query = AsyncMock(return_value=mock_result)
        
        query = "I want to visit Rome - what's the weather like?"
        result = await langgraph_agent.process_query(query)
        
        assert result.route_taken == "combined"
        assert len(result.sources) == 2
        assert len(result.weather_data) == 1
        assert len(result.locations) == 1

    @pytest.mark.asyncio
    async def test_process_query_error(self, langgraph_agent):
        """Test query processing error handling."""
        langgraph_agent.rag_graph.process_query = AsyncMock(side_effect=Exception("Processing failed"))
        
        query = "What's the weather in Rome?"
        result = await langgraph_agent.process_query(query)
        
        assert "error" in result.answer.lower()
        assert result.route_taken == "error"
        assert result.error == "Processing failed"
        assert result.sources == []
        assert result.weather_data == []
        assert result.locations == []

    def test_build_agent_response_success(self, langgraph_agent):
        """Test successful agent response building."""
        result_data = {
            "final_answer": "This is a test answer.",
            "route": "weather",
            "sources": [
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
        
        processing_time = 150.5
        response = langgraph_agent._build_agent_response(result_data, processing_time)
        
        assert isinstance(response, AgentResponse)
        assert response.answer == "This is a test answer."
        assert response.route_taken == "weather"
        assert response.processing_time_ms == processing_time
        assert len(response.weather_data) == 1
        assert len(response.locations) == 1
        assert response.error is None

    def test_build_agent_response_with_answer_field(self, langgraph_agent):
        """Test agent response building with 'answer' field instead of 'final_answer'."""
        result_data = {
            "answer": "This is a test answer.",
            "route": "document",
            "sources": []
        }
        
        processing_time = 100.0
        response = langgraph_agent._build_agent_response(result_data, processing_time)
        
        assert response.answer == "This is a test answer."
        assert response.route_taken == "document"

    def test_build_agent_response_error(self, langgraph_agent):
        """Test agent response building error handling."""
        result_data = {
            "final_answer": "This is a test answer.",
            "route": "weather",
            "sources": "invalid_sources"  # This should cause an error
        }
        
        processing_time = 100.0
        
        # Mock the error handling
        with patch('traceback.print_exc'):
            response = langgraph_agent._build_agent_response(result_data, processing_time)
            
            assert "error" in response.answer.lower()
            assert response.route_taken == "error"

    def test_build_error_response(self, langgraph_agent):
        """Test error response building."""
        query = "Test query"
        error = "Test error"
        processing_time = 200.0
        
        response = langgraph_agent._build_error_response(query, error, processing_time)
        
        assert isinstance(response, AgentResponse)
        assert "error" in response.answer.lower()
        assert "Test query" in response.answer
        assert response.route_taken == "error"
        assert response.error == error
        assert response.processing_time_ms == processing_time
        assert response.sources == []
        assert response.weather_data == []
        assert response.locations == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, langgraph_agent):
        """Test successful health check."""
        langgraph_agent.rag_graph.health_check.return_value = True
        
        result = await langgraph_agent.health_check()
        
        assert result["rag_graph"] is True
        assert result["modular_agents"] is True

    @pytest.mark.asyncio
    async def test_health_check_rag_graph_unhealthy(self, langgraph_agent):
        """Test health check with unhealthy RAG graph."""
        langgraph_agent.rag_graph.health_check.return_value = False
        
        result = await langgraph_agent.health_check()
        
        assert result["rag_graph"] is False
        assert result["modular_agents"] is True

    def test_weather_data_parsing(self, langgraph_agent):
        """Test weather data parsing from sources."""
        result_data = {
            "final_answer": "Test answer",
            "route": "weather",
            "sources": [
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
        }
        
        processing_time = 100.0
        response = langgraph_agent._build_agent_response(result_data, processing_time)
        
        assert len(response.weather_data) == 1
        weather = response.weather_data[0]
        assert weather.city == "Rome"
        assert weather.country == "IT"
        assert weather.temperature["current"] == 22.5
        assert weather.conditions["description"] == "Partly cloudy"
        assert weather.humidity == 65
        assert weather.pressure == 1013
        assert weather.wind["speed"] == 3.2
        assert weather.visibility == 10.0
        assert weather.cloudiness == 40
        assert weather.timestamp == 1640995200

    def test_location_info_parsing(self, langgraph_agent):
        """Test location info parsing from sources."""
        result_data = {
            "final_answer": "Test answer",
            "route": "weather",
            "sources": [
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
        
        processing_time = 100.0
        response = langgraph_agent._build_agent_response(result_data, processing_time)
        
        assert len(response.locations) == 1
        location = response.locations[0]
        assert location.name == "Rome"
        assert location.context == []
        assert location.weather_data is None  # Weather data is stored separately
