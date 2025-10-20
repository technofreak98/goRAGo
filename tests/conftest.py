"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models import Document, Chunk, DocumentStructure, DocumentMetadata, SearchResult, SearchQuery
from app.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_elasticsearch_client():
    """Mock Elasticsearch client."""
    mock_client = Mock()
    mock_client.indices = Mock()
    mock_client.indices.exists = Mock(return_value=False)
    mock_client.indices.create = Mock()
    mock_client.search = Mock()
    mock_client.get = Mock()
    mock_client.index = Mock()
    mock_client.delete = Mock()
    mock_client.delete_by_query = Mock()
    mock_client.cluster = Mock()
    mock_client.cluster.health = Mock(return_value={"status": "green"})
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.embeddings = Mock()
    mock_client.embeddings.create = Mock()
    
    # Mock response objects with proper token counting
    def create_mock_embedding_response(embeddings):
        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in embeddings]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = len(embeddings) * 10  # Mock token count
        return mock_response
    
    def create_mock_chat_response(content, prompt_tokens=100, completion_tokens=50):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = content
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens
        mock_response.usage.total_tokens = prompt_tokens + completion_tokens
        return mock_response
    
    mock_client.embeddings.create.return_value = create_mock_embedding_response([[0.1] * 1536])
    mock_client.chat.completions.create.return_value = create_mock_chat_response("Test response")
    
    return mock_client


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    metadata = DocumentMetadata(
        title="Test Book",
        author="Test Author",
        language="en"
    )
    
    structure = DocumentStructure(
        metadata=metadata,
        parts=[],
        chapters=[],
        sections=[]
    )
    
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            text="This is a test chunk about Rome and Italy.",
            token_count=10,
            document_id="doc_1",
            level=0,
            embedding=[0.1] * 1536
        ),
        Chunk(
            chunk_id="chunk_2", 
            text="Another chunk about Venice and canals.",
            token_count=8,
            document_id="doc_1",
            level=0,
            embedding=[0.2] * 1536
        )
    ]
    
    return Document(
        document_id="doc_1",
        structure=structure,
        chunks=chunks,
        total_chunks=2
    )


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            chunk_id="chunk_1",
            text="This is a test chunk about Rome and Italy.",
            score=0.95,
            document_id="doc_1",
            level=0,
            rank=1,
            relevance_score=0.95,
            token_count=10
        ),
        SearchResult(
            chunk_id="chunk_2",
            text="Another chunk about Venice and canals.",
            score=0.87,
            document_id="doc_1", 
            level=0,
            rank=2,
            relevance_score=0.87,
            token_count=8
        )
    ]


@pytest.fixture
def sample_search_query():
    """Sample search query for testing."""
    return SearchQuery(
        query="Rome Italy travel",
        top_k=10,
        rerank=True,
        compression=True
    )


@pytest.fixture
def mock_weather_data():
    """Mock weather data for testing."""
    return [
        {
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


@pytest.fixture
def mock_agent_state():
    """Mock agent state for testing."""
    return {
        "query": "What's the weather in Rome?",
        "route": "weather",
        "confidence": 0.9,
        "reasoning": "Query asks for weather information",
        "document_context": "",
        "weather_context": "",
        "document_sources": [],
        "weather_sources": [],
        "extracted_places": ["Rome"],
        "final_answer": "",
        "sources": []
    }


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return {
        "openai_api_key": "test_key",
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.7,
        "elasticsearch_url": "http://localhost:9200",
        "embedding_model": "text-embedding-3-small",
        "embedding_batch_size": 100,
        "child_index_name": "test_child",
        "openweather_api_key": "test_weather_key",
        "initial_top_k": 20,
        "final_top_k": 10,
        "max_tokens_compression": 1500
    }


@pytest.fixture(autouse=True)
def setup_test_environment(mock_settings):
    """Set up test environment with mocked settings."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "OPENWEATHER_API_KEY": "test_weather_key"
    }):
        yield
