"""Elasticsearch service for managing vector database operations."""

import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from app.config import settings
from app.models import Chunk, Document, SearchResult, SectionInfo, ChapterInfo, PartInfo

logger = logging.getLogger(__name__)


class ElasticsearchService:
    """Service for Elasticsearch operations with document chunking."""
    
    def __init__(self):
        """Initialize Elasticsearch client."""
        self.client = self._create_client()
        self.child_index = settings.child_index_name
        
    def _create_client(self) -> Elasticsearch:
        """Create Elasticsearch client with authentication if needed."""
        auth = None
        if settings.elasticsearch_username and settings.elasticsearch_password:
            auth = (settings.elasticsearch_username, settings.elasticsearch_password)
            
        return Elasticsearch(
            hosts=[settings.elasticsearch_url],
            http_auth=auth,
            verify_certs=False,
            request_timeout=60
        )
    
    async def create_indices(self) -> bool:
        """Create child index with proper mappings."""
        try:
            
            # Create child index mapping
            child_mapping = {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "text": {"type": "text", "analyzer": "standard"},
                        "token_count": {"type": "integer"},
                        "parent_window": {"type": "text", "analyzer": "standard"},
                        "document_id": {"type": "keyword"},
                        "level": {"type": "integer"},
                        "parent_id": {"type": "keyword"},
                        "child_ids": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1536,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "section_info": {
                            "type": "object",
                            "properties": {
                                "section_number": {"type": "integer"},
                                "title": {"type": "text"},
                                "chapter_number": {"type": "integer"},
                                "part_number": {"type": "integer"},
                                "start_page": {"type": "integer"}
                            }
                        },
                        "chapter_info": {
                            "type": "object",
                            "properties": {
                                "chapter_number": {"type": "integer"},
                                "title": {"type": "text"},
                                "part_number": {"type": "integer"},
                                "start_page": {"type": "integer"}
                            }
                        },
                        "part_info": {
                            "type": "object",
                            "properties": {
                                "part_number": {"type": "integer"},
                                "title": {"type": "text"},
                                "start_page": {"type": "integer"}
                            }
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            # Create child index
            if not self.client.indices.exists(index=self.child_index):
                self.client.indices.create(index=self.child_index, body=child_mapping)
                logger.info(f"Created child index: {self.child_index}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating indices: {e}")
            return False
    
    async def index_child_chunks(self, document: Document) -> bool:
        """Index child chunks for a document."""
        try:
            for i, chunk in enumerate(document.chunks):
                try:
                    doc_body = {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "parent_window": chunk.parent_window or "",
                        "document_id": chunk.document_id,
                        "level": chunk.level,
                        "parent_id": chunk.parent_id,
                        "child_ids": chunk.child_ids,
                        "embedding": chunk.embedding,
                        "section_info": chunk.section_info.model_dump() if chunk.section_info else {},
                        "chapter_info": chunk.chapter_info.model_dump() if chunk.chapter_info else {},
                        "part_info": chunk.part_info.model_dump() if chunk.part_info else {}
                    }
                    
                    response = self.client.index(
                        index=self.child_index,
                        id=chunk.chunk_id,
                        document=doc_body,
                        routing=document.document_id
                    )
                    
                    logger.debug(f"Indexed chunk {i+1}/{len(document.chunks)}: {chunk.chunk_id}")
                    
                except Exception as chunk_error:
                    logger.error(f"Error indexing chunk {i+1} ({chunk.chunk_id}): {chunk_error}")
                    raise chunk_error
            
            logger.info(f"Indexed {len(document.chunks)} chunks for document: {document.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing child chunks: {e}")
            return False
    
    async def hybrid_search(self, query: str, query_embedding: List[float], 
                          top_k: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """Perform hybrid search combining dense vector and BM25."""
        try:
            # Build filter query
            filter_query = []
            if filters:
                if filters.get("part_number"):
                    filter_query.append({
                        "term": {"part_info.part_number": filters["part_number"]}
                    })
                if filters.get("chapter_number"):
                    filter_query.append({
                        "term": {"chapter_info.chapter_number": filters["chapter_number"]}
                    })
            
            # BM25 keyword search
            bm25_query = {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "parent_window"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            
            # Dense vector search using script_score (compatible with older Elasticsearch versions)
            vector_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
            
            # Combined query using bool with should clauses for hybrid search
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": bm25_query,
                                    "script": {
                                        "source": "_score"
                                    }
                                }
                            },
                            vector_query
                        ],
                        "filter": filter_query if filter_query else None,
                        "minimum_should_match": 1
                    }
                }
            }
            
            # Remove None values
            if not search_body["query"]["bool"]["filter"]:
                del search_body["query"]["bool"]["filter"]
            
            response = self.client.search(
                index=self.child_index,
                body=search_body
            )
            
            results = []
            for rank, hit in enumerate(response["hits"]["hits"], 1):
                source = hit["_source"]
                
                # Create combined context for RAG
                context_parts = [source["text"]]
                if source.get("parent_window"):
                    context_parts.append(f"Context: {source['parent_window']}")
                combined_context = "\n\n".join(context_parts)
                
                # Calculate normalized relevance score (0-1)
                max_possible_score = 100.0  # Adjust based on your scoring system
                normalized_score = min(hit["_score"] / max_possible_score, 1.0)
                
                # Parse structure info safely
                section_info = None
                chapter_info = None
                part_info = None
                
                # Parse section_info if it exists and has required fields
                if source.get("section_info") and isinstance(source["section_info"], dict):
                    section_data = source["section_info"]
                    if all(key in section_data for key in ["section_number", "title", "chapter_number"]):
                        section_info = SectionInfo(
                            section_number=section_data["section_number"],
                            title=section_data["title"],
                            chapter_number=section_data["chapter_number"],
                            part_number=section_data.get("part_number"),
                            start_page=section_data.get("start_page")
                        )
                
                # Parse chapter_info if it exists and has required fields
                if source.get("chapter_info") and isinstance(source["chapter_info"], dict):
                    chapter_data = source["chapter_info"]
                    if all(key in chapter_data for key in ["chapter_number", "title"]):
                        chapter_info = ChapterInfo(
                            chapter_number=chapter_data["chapter_number"],
                            title=chapter_data["title"],
                            part_number=chapter_data.get("part_number"),
                            start_page=chapter_data.get("start_page")
                        )
                
                # Parse part_info if it exists and has required fields
                if source.get("part_info") and isinstance(source["part_info"], dict):
                    part_data = source["part_info"]
                    if all(key in part_data for key in ["part_number", "title"]):
                        part_info = PartInfo(
                            part_number=part_data["part_number"],
                            title=part_data["title"],
                            start_page=part_data.get("start_page")
                        )
                
                results.append(SearchResult(
                    chunk_id=source["chunk_id"],
                    text=source["text"],
                    score=hit["_score"],
                    document_id=source.get("document_id", "unknown"),
                    level=source.get("level", 0),
                    parent_id=source.get("parent_id"),
                    child_ids=source.get("child_ids", []),
                    rank=rank,
                    parent_window=source.get("parent_window"),
                    section_info=section_info,
                    chapter_info=chapter_info,
                    part_info=part_info,
                    # RAG-specific fields
                    context=combined_context,
                    relevance_score=normalized_score,
                    token_count=source.get("token_count", 0)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk with parent context."""
        try:
            response = self.client.get(
                index=self.child_index,
                id=chunk_id
            )
            
            return response["_source"]
            
        except NotFoundError:
            logger.warning(f"Chunk not found: {chunk_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk: {e}")
            return None
    
    async def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status by checking if chunks exist."""
        try:
            response = self.client.search(
                index=self.child_index,
                body={
                    "size": 1,
                    "query": {
                        "term": {
                            "document_id": document_id
                        }
                    }
                }
            )
            
            if response["hits"]["total"]["value"] > 0:
                # Document exists, return basic status info
                return {
                    "document_id": document_id,
                    "status": "indexed",
                    "chunk_count": response["hits"]["total"]["value"]
                }
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error getting document status: {e}")
            return None
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        try:
            response = self.client.search(
                index=self.child_index,
                body={
                    "size": 10,
                    "query": {"match_all": {}},
                    "sort": [{"chunk_id": {"order": "asc"}}]
                }
            )
            
            documents = []
            for hit in response["hits"]["hits"]:
                documents.append(hit["_source"])
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def _extract_full_text(self, document: Document) -> str:
        """Extract full text from document chunks."""
        return " ".join([chunk.text for chunk in document.chunks])
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Delete all chunks for this document
            self.client.delete_by_query(
                index=self.child_index,
                body={
                    "query": {
                        "term": {
                            "document_id": document_id
                        }
                    }
                }
            )
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
