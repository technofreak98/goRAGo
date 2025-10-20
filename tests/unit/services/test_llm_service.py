"""Unit tests for LLMService."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import openai

from app.services.llm_service import LLMService


@pytest.mark.unit
class TestLLMService:
    """Test cases for LLMService."""

    @pytest.fixture
    def llm_service(self, mock_openai_client):
        """Create LLMService instance with mocked client."""
        with patch('app.services.llm_service.openai.OpenAI') as mock_openai:
            mock_openai.return_value = mock_openai_client
            service = LLMService()
            service.client = mock_openai_client
            return service

    @pytest.mark.asyncio
    async def test_generate_answer_success(self, llm_service, mock_openai_client):
        """Test successful answer generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test answer about Rome and Italy."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 75
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        query = "Tell me about Rome"
        context = "Rome is the capital of Italy and has many historical sites."
        
        result = await llm_service.generate_answer(query, context)
        
        assert "Rome" in result
        assert "Italy" in result
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_with_weather(self, llm_service, mock_openai_client, mock_weather_data):
        """Test answer generation with weather data."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Rome is beautiful and currently has partly cloudy weather."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 60
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 90
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        query = "What's the weather like in Rome?"
        context = "Rome is the capital of Italy."
        
        result = await llm_service.generate_answer(query, context, mock_weather_data)
        
        assert "Rome" in result
        assert "weather" in result.lower()
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_error(self, llm_service, mock_openai_client):
        """Test answer generation error handling."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        query = "Tell me about Rome"
        context = "Rome is the capital of Italy."
        
        result = await llm_service.generate_answer(query, context)
        
        assert "apologize" in result.lower()
        assert "trouble" in result.lower()

    @pytest.mark.asyncio
    async def test_classify_query_intent_document(self, llm_service, mock_openai_client):
        """Test query intent classification for document queries."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "document"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 45
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        query = "What places did Mark Twain visit?"
        
        result = await llm_service.classify_query_intent(query)
        
        assert result == "document"

    @pytest.mark.asyncio
    async def test_classify_query_intent_weather(self, llm_service, mock_openai_client):
        """Test query intent classification for weather queries."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "weather"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 45
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        query = "What's the weather in Rome?"
        
        result = await llm_service.classify_query_intent(query)
        
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_classify_query_intent_combined(self, llm_service, mock_openai_client):
        """Test query intent classification for combined queries."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "combined"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 55
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        query = "I want to visit places Twain went to in Italy - what's the weather?"
        
        result = await llm_service.classify_query_intent(query)
        
        assert result == "combined"

    @pytest.mark.asyncio
    async def test_classify_query_intent_guardrails(self, llm_service, mock_openai_client):
        """Test query intent classification for out-of-scope queries."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "guardrails"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 45
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        query = "Explain quantum physics"
        
        result = await llm_service.classify_query_intent(query)
        
        assert result == "guardrails"

    @pytest.mark.asyncio
    async def test_classify_query_intent_error(self, llm_service, mock_openai_client):
        """Test query intent classification error handling."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        query = "What's the weather in Rome?"
        
        result = await llm_service.classify_query_intent(query)
        
        assert result == "guardrails"  # Default fallback

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, llm_service, mock_openai_client):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test response."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = await llm_service.chat_completion(messages)
        
        assert result == "This is a test response."
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_error(self, llm_service, mock_openai_client):
        """Test chat completion error handling."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception):
            await llm_service.chat_completion(messages)

    def test_parse_classification_document(self, llm_service):
        """Test classification parsing for document."""
        result = llm_service._parse_classification("document")
        assert result == "document"

    def test_parse_classification_weather(self, llm_service):
        """Test classification parsing for weather."""
        result = llm_service._parse_classification("weather")
        assert result == "weather"

    def test_parse_classification_combined(self, llm_service):
        """Test classification parsing for combined."""
        result = llm_service._parse_classification("combined")
        assert result == "combined"

    def test_parse_classification_guardrails(self, llm_service):
        """Test classification parsing for guardrails."""
        result = llm_service._parse_classification("guardrails")
        assert result == "guardrails"

    def test_parse_classification_unknown(self, llm_service):
        """Test classification parsing for unknown response."""
        result = llm_service._parse_classification("unknown response")
        assert result == "guardrails"  # Default fallback

    def test_format_weather_for_prompt(self, llm_service, mock_weather_data):
        """Test weather data formatting for prompt."""
        result = llm_service._format_weather_for_prompt(mock_weather_data)
        
        assert "Rome" in result
        assert "22.5°C" in result
        assert "Partly cloudy" in result
        assert "65%" in result

    def test_format_weather_for_prompt_empty(self, llm_service):
        """Test weather data formatting with empty data."""
        result = llm_service._format_weather_for_prompt([])
        
        assert result == "No weather data available."

    def test_get_fallback_response_with_weather(self, llm_service, mock_weather_data):
        """Test fallback response with weather data."""
        result = llm_service._get_fallback_response("Test query", mock_weather_data)
        
        assert "apologize" in result.lower()
        assert "Rome" in result
        assert "weather" in result.lower()

    def test_get_fallback_response_without_weather(self, llm_service):
        """Test fallback response without weather data."""
        result = llm_service._get_fallback_response("Test query", None)
        
        assert "apologize" in result.lower()
        assert "literature" in result.lower()

    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service, mock_openai_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await llm_service.health_check()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service, mock_openai_client):
        """Test health check failure."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = await llm_service.health_check()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_extract_places_from_text_success(self, llm_service, mock_openai_client):
        """Test successful place extraction."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Rome\nVenice\nFlorence"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 45
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        text = "I visited Rome, Venice, and Florence during my trip to Italy."
        
        result = await llm_service.extract_places_from_text(text)
        
        assert "Rome" in result
        assert "Venice" in result
        assert "Florence" in result

    @pytest.mark.asyncio
    async def test_extract_places_from_text_empty(self, llm_service, mock_openai_client):
        """Test place extraction with empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 20
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        text = "This text has no places."
        
        result = await llm_service.extract_places_from_text(text)
        
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_places_from_text_error(self, llm_service, mock_openai_client):
        """Test place extraction error handling."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        text = "I visited Rome and Venice."
        
        result = await llm_service.extract_places_from_text(text)
        
        assert result == []

    def test_get_model_info(self, llm_service):
        """Test model information retrieval."""
        info = llm_service.get_model_info()
        
        assert "model" in info
        assert "temperature" in info
        assert "max_tokens" in info
        assert info["model"] == llm_service.model
        assert info["temperature"] == llm_service.temperature
        assert info["max_tokens"] == llm_service.max_tokens
