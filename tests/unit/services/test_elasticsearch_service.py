"""Unit tests for ElasticsearchService."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from elasticsearch.exceptions import NotFoundError, RequestError

from app.services.elasticsearch_service import ElasticsearchService
from app.models import Document, Chunk, DocumentStructure, DocumentMetadata, SearchResult


@pytest.mark.unit
class TestElasticsearchService:
    """Test cases for ElasticsearchService."""

    @pytest.fixture
    def es_service(self, mock_elasticsearch_client):
        """Create ElasticsearchService instance with mocked client."""
        with patch('app.services.elasticsearch_service.Elasticsearch') as mock_es:
            mock_es.return_value = mock_elasticsearch_client
            service = ElasticsearchService()
            service.client = mock_elasticsearch_client
            return service

    @pytest.mark.asyncio
    async def test_create_indices_success(self, es_service, mock_elasticsearch_client):
        """Test successful index creation."""
        mock_elasticsearch_client.indices.exists.return_value = False
        mock_elasticsearch_client.indices.create.return_value = {"acknowledged": True}
        
        result = await es_service.create_indices()
        
        assert result is True
        assert mock_elasticsearch_client.indices.create.call_count == 1  # child index only

    @pytest.mark.asyncio
    async def test_create_indices_existing(self, es_service, mock_elasticsearch_client):
        """Test index creation when indices already exist."""
        mock_elasticsearch_client.indices.exists.return_value = True
        
        result = await es_service.create_indices()
        
        assert result is True
        mock_elasticsearch_client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_indices_error(self, es_service, mock_elasticsearch_client):
        """Test index creation error handling."""
        mock_elasticsearch_client.indices.exists.return_value = False
        mock_meta = Mock()
        mock_meta.status = 500
        mock_elasticsearch_client.indices.create.side_effect = RequestError(
            message="Index creation failed",
            meta=mock_meta,
            body={"error": {"type": "index_creation_failed"}}
        )
        
        result = await es_service.create_indices()
        
        assert result is False


    @pytest.mark.asyncio
    async def test_index_child_chunks_success(self, es_service, mock_elasticsearch_client, sample_document):
        """Test successful child chunks indexing."""
        mock_elasticsearch_client.index.return_value = {"result": "created"}
        
        result = await es_service.index_child_chunks(sample_document)
        
        assert result is True
        assert mock_elasticsearch_client.index.call_count == len(sample_document.chunks)

    @pytest.mark.asyncio
    async def test_index_child_chunks_error(self, es_service, mock_elasticsearch_client, sample_document):
        """Test child chunks indexing error handling."""
        mock_meta = Mock()
        mock_meta.status = 500
        mock_elasticsearch_client.index.side_effect = RequestError(
            message="Indexing failed",
            meta=mock_meta,
            body={"error": {"type": "indexing_failed"}}
        )
        
        result = await es_service.index_child_chunks(sample_document)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, es_service, mock_elasticsearch_client):
        """Test successful hybrid search."""
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "chunk_id": "chunk_1",
                            "text": "Test text about Rome",
                            "parent_window": "Context about Italy",
                            "document_id": "doc_1",
                            "level": 0,
                            "parent_id": None,
                            "child_ids": [],
                            "token_count": 10,
                            "section_info": {},
                            "chapter_info": {},
                            "part_info": {}
                        },
                        "_score": 0.95
                    }
                ]
            }
        }
        mock_elasticsearch_client.search.return_value = mock_response
        
        query = "Rome Italy"
        query_embedding = [0.1] * 1536
        
        results = await es_service.hybrid_search(query, query_embedding, top_k=10)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == "Test text about Rome"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, es_service, mock_elasticsearch_client):
        """Test hybrid search with filters."""
        mock_response = {"hits": {"hits": []}}
        mock_elasticsearch_client.search.return_value = mock_response
        
        query = "Rome Italy"
        query_embedding = [0.1] * 1536
        filters = {"part_number": 1, "chapter_number": 2}
        
        results = await es_service.hybrid_search(query, query_embedding, top_k=10, filters=filters)
        
        assert len(results) == 0
        mock_elasticsearch_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_error(self, es_service, mock_elasticsearch_client):
        """Test hybrid search error handling."""
        mock_meta = Mock()
        mock_meta.status = 500
        mock_elasticsearch_client.search.side_effect = RequestError(
            message="Search failed",
            meta=mock_meta,
            body={"error": {"type": "search_failed"}}
        )
        
        query = "Rome Italy"
        query_embedding = [0.1] * 1536
        
        results = await es_service.hybrid_search(query, query_embedding, top_k=10)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_get_chunk_success(self, es_service, mock_elasticsearch_client):
        """Test successful chunk retrieval."""
        mock_response = {
            "_source": {
                "chunk_id": "chunk_1",
                "text": "Test chunk text",
                "document_id": "doc_1"
            }
        }
        mock_elasticsearch_client.get.return_value = mock_response
        
        result = await es_service.get_chunk("chunk_1")
        
        assert result == mock_response["_source"]
        mock_elasticsearch_client.get.assert_called_once_with(
            index=es_service.child_index,
            id="chunk_1"
        )

    @pytest.mark.asyncio
    async def test_get_chunk_not_found(self, es_service, mock_elasticsearch_client):
        """Test chunk retrieval when chunk not found."""
        mock_meta = Mock()
        mock_meta.status = 404
        mock_elasticsearch_client.get.side_effect = NotFoundError(
            message="Chunk not found",
            meta=mock_meta,
            body={"error": {"type": "not_found"}}
        )
        
        result = await es_service.get_chunk("nonexistent_chunk")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_status_success(self, es_service, mock_elasticsearch_client):
        """Test successful document status retrieval."""
        mock_response = {
            "hits": {
                "total": {"value": 3},
                "hits": [{"_source": {"document_id": "doc_1"}}]
            }
        }
        mock_elasticsearch_client.search.return_value = mock_response
        
        result = await es_service.get_document_status("doc_1")
        
        expected = {
            "document_id": "doc_1",
            "status": "indexed",
            "chunk_count": 3
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_document_status_not_found(self, es_service, mock_elasticsearch_client):
        """Test document status retrieval when document not found."""
        mock_response = {
            "hits": {
                "total": {"value": 0},
                "hits": []
            }
        }
        mock_elasticsearch_client.search.return_value = mock_response
        
        result = await es_service.get_document_status("nonexistent_doc")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_list_documents_success(self, es_service, mock_elasticsearch_client):
        """Test successful document listing."""
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "document_id": "doc_1",
                            "title": "Test Document 1"
                        }
                    },
                    {
                        "_source": {
                            "document_id": "doc_2", 
                            "title": "Test Document 2"
                        }
                    }
                ]
            }
        }
        mock_elasticsearch_client.search.return_value = mock_response
        
        result = await es_service.list_documents()
        
        assert len(result) == 2
        assert result[0]["document_id"] == "doc_1"
        assert result[1]["document_id"] == "doc_2"

    @pytest.mark.asyncio
    async def test_list_documents_error(self, es_service, mock_elasticsearch_client):
        """Test document listing error handling."""
        mock_meta = Mock()
        mock_meta.status = 500
        mock_elasticsearch_client.search.side_effect = RequestError(
            message="Search failed",
            meta=mock_meta,
            body={"error": {"type": "search_failed"}}
        )
        
        result = await es_service.list_documents()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_document_success(self, es_service, mock_elasticsearch_client):
        """Test successful document deletion."""
        mock_elasticsearch_client.delete_by_query.return_value = {"deleted": 5}
        
        result = await es_service.delete_document("doc_1")
        
        assert result is True
        mock_elasticsearch_client.delete_by_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_error(self, es_service, mock_elasticsearch_client):
        """Test document deletion error handling."""
        mock_meta = Mock()
        mock_meta.status = 500
        mock_elasticsearch_client.delete_by_query.side_effect = RequestError(
            message="Delete failed",
            meta=mock_meta,
            body={"error": {"type": "delete_failed"}}
        )
        
        result = await es_service.delete_document("doc_1")
        
        assert result is False

    def test_extract_full_text(self, es_service, sample_document):
        """Test full text extraction from document."""
        full_text = es_service._extract_full_text(sample_document)
        
        expected_text = " ".join([chunk.text for chunk in sample_document.chunks])
        assert full_text == expected_text
