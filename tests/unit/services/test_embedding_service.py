"""Unit tests for EmbeddingService."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import openai

from app.services.embedding_service import EmbeddingService


@pytest.mark.unit
class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture
    def embedding_service(self, mock_openai_client):
        """Create EmbeddingService instance with mocked client."""
        with patch('app.services.embedding_service.openai.OpenAI') as mock_openai:
            mock_openai.return_value = mock_openai_client
            service = EmbeddingService()
            service.client = mock_openai_client
            return service

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service, mock_openai_client):
        """Test successful batch embedding generation."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
            Mock(embedding=[0.3] * 1536)
        ]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 30  # 3 texts * 10 tokens each
        mock_openai_client.embeddings.create.return_value = mock_response
        
        texts = ["Text 1", "Text 2", "Text 3"]
        
        result = await embedding_service.generate_embeddings(texts)
        
        assert len(result) == 3
        assert all(len(emb) == 1536 for emb in result)
        assert result[0] == [0.1] * 1536
        assert result[1] == [0.2] * 1536
        assert result[2] == [0.3] * 1536
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_large_batch(self, embedding_service, mock_openai_client):
        """Test embedding generation with large batch that requires multiple API calls."""
        def create_mock_response(batch_size):
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(batch_size)]
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = batch_size * 10
            return mock_response
        
        # Mock responses for 3 batches: 100, 100, 50
        mock_openai_client.embeddings.create.side_effect = [
            create_mock_response(100),
            create_mock_response(100), 
            create_mock_response(50)
        ]
        
        # Create a large batch that exceeds batch_size
        texts = [f"Text {i}" for i in range(250)]  # 250 texts, batch_size is 100
        
        result = await embedding_service.generate_embeddings(texts)
        
        assert len(result) == 250
        assert mock_openai_client.embeddings.create.call_count == 3  # 3 batches: 100, 100, 50

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, embedding_service, mock_openai_client):
        """Test embedding generation error handling."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        texts = ["Text 1", "Text 2"]
        
        with pytest.raises(Exception):
            await embedding_service.generate_embeddings(texts)

    @pytest.mark.asyncio
    async def test_generate_single_embedding_success(self, embedding_service, mock_openai_client):
        """Test successful single embedding generation."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 10  # Single text
        mock_openai_client.embeddings.create.return_value = mock_response
        
        text = "Single text for embedding"
        
        result = await embedding_service.generate_single_embedding(text)
        
        assert result == [0.1] * 1536
        mock_openai_client.embeddings.create.assert_called_once_with(
            model=embedding_service.model,
            input=text
        )

    @pytest.mark.asyncio
    async def test_generate_single_embedding_error(self, embedding_service, mock_openai_client):
        """Test single embedding generation error handling."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        text = "Single text for embedding"
        
        with pytest.raises(Exception):
            await embedding_service.generate_single_embedding(text)

    @pytest.mark.asyncio
    async def test_generate_query_embedding_success(self, embedding_service, mock_openai_client):
        """Test successful query embedding generation."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 10  # Single query
        mock_openai_client.embeddings.create.return_value = mock_response
        
        # Mock query preprocessor
        with patch('app.services.query_preprocessor.QueryPreprocessor') as mock_preprocessor:
            mock_preprocessor.return_value.get_embedding_query.return_value = "processed query"
            
            query = "Rome Italy travel"
            
            result = await embedding_service.generate_query_embedding(query)
            
            assert result == [0.1] * 1536
            mock_preprocessor.return_value.get_embedding_query.assert_called_once_with(query)
            mock_openai_client.embeddings.create.assert_called_once_with(
                model=embedding_service.model,
                input="processed query"
            )

    @pytest.mark.asyncio
    async def test_generate_query_embedding_error(self, embedding_service, mock_openai_client):
        """Test query embedding generation error handling."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        query = "Rome Italy travel"
        
        with pytest.raises(Exception):
            await embedding_service.generate_query_embedding(query)

    def test_clean_text(self, embedding_service):
        """Test text cleaning for embedding generation."""
        dirty_text = "  This   is  a  test  text  with  extra  spaces!  @#$%^&*()  "
        
        cleaned = embedding_service._clean_text(dirty_text)
        
        assert cleaned == "This is a test text with extra spaces!"
        assert "  " not in cleaned  # No double spaces
        assert cleaned.strip() == cleaned  # No leading/trailing spaces

    def test_clean_text_special_characters(self, embedding_service):
        """Test text cleaning with special characters."""
        text_with_special = "Text with special chars: @#$%^&*()[]{}|\\:;\"'<>?,./"
        
        cleaned = embedding_service._clean_text(text_with_special)
        
        # Should remove most special characters but keep common punctuation
        assert "@#$%^&*()[]{}|\\" not in cleaned
        assert "Text with special chars" in cleaned

    @pytest.mark.asyncio
    async def test_batch_embed_chunks_success(self, embedding_service, mock_openai_client):
        """Test successful batch embedding of chunks."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536)
        ]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 20  # 2 chunks * 10 tokens each
        mock_openai_client.embeddings.create.return_value = mock_response
        
        chunks = [
            {"text": "Chunk 1", "chunk_id": "chunk_1"},
            {"text": "Chunk 2", "chunk_id": "chunk_2"}
        ]
        
        result = await embedding_service.batch_embed_chunks(chunks)
        
        assert len(result) == 2
        assert result[0]["embedding"] == [0.1] * 1536
        assert result[1]["embedding"] == [0.2] * 1536
        assert result[0]["chunk_id"] == "chunk_1"
        assert result[1]["chunk_id"] == "chunk_2"

    @pytest.mark.asyncio
    async def test_batch_embed_chunks_error(self, embedding_service, mock_openai_client):
        """Test batch embedding error handling."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        chunks = [{"text": "Chunk 1", "chunk_id": "chunk_1"}]
        
        with pytest.raises(Exception):
            await embedding_service.batch_embed_chunks(chunks)

    def test_get_embedding_dimensions_3_small(self, embedding_service):
        """Test embedding dimensions for text-embedding-3-small."""
        embedding_service.model = "text-embedding-3-small"
        
        dimensions = embedding_service.get_embedding_dimensions()
        
        assert dimensions == 1536

    def test_get_embedding_dimensions_3_large(self, embedding_service):
        """Test embedding dimensions for text-embedding-3-large."""
        embedding_service.model = "text-embedding-3-large"
        
        dimensions = embedding_service.get_embedding_dimensions()
        
        assert dimensions == 3072

    def test_get_embedding_dimensions_ada_002(self, embedding_service):
        """Test embedding dimensions for text-embedding-ada-002."""
        embedding_service.model = "text-embedding-ada-002"
        
        dimensions = embedding_service.get_embedding_dimensions()
        
        assert dimensions == 1536

    def test_get_embedding_dimensions_default(self, embedding_service):
        """Test embedding dimensions for unknown model (default)."""
        embedding_service.model = "unknown-model"
        
        dimensions = embedding_service.get_embedding_dimensions()
        
        assert dimensions == 1536  # Default

    @pytest.mark.asyncio
    async def test_health_check_success(self, embedding_service, mock_openai_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 10  # Single test text
        mock_openai_client.embeddings.create.return_value = mock_response
        
        result = await embedding_service.health_check()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_wrong_dimensions(self, embedding_service, mock_openai_client):
        """Test health check with wrong embedding dimensions."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1000)]  # Wrong dimensions
        mock_openai_client.embeddings.create.return_value = mock_response
        
        result = await embedding_service.health_check()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_error(self, embedding_service, mock_openai_client):
        """Test health check error handling."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        result = await embedding_service.health_check()
        
        assert result is False
