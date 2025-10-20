"""Search router for document retrieval and search operations."""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models import SearchQuery, SearchResponse, ChunkResponse
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


class SimpleSearchQuery(BaseModel):
    """Simple search query model for the API."""
    query: str
    top_k: int = 10
    rerank: bool = True
    compression: bool = True
    filter_by_part: Optional[int] = None
    filter_by_chapter: Optional[int] = None


@router.post("/search", response_model=SearchResponse)
async def search_documents(search_query: SimpleSearchQuery):
    """Search documents using hybrid retrieval."""
    try:
        # Convert to SearchQuery model
        query = SearchQuery(
            query=search_query.query,
            top_k=search_query.top_k,
            rerank=search_query.rerank,
            compression=search_query.compression,
            filter_by_part=search_query.filter_by_part,
            filter_by_chapter=search_query.filter_by_chapter
        )
        
        # Perform search
        retrieval_service = RetrievalService()
        results = await retrieval_service.search(query)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/chunk/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str):
    """Retrieve a specific chunk with full parent context."""
    try:
        retrieval_service = RetrievalService()
        chunk_data = await retrieval_service.get_chunk_with_context(chunk_id)
        
        if not chunk_data:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        # Convert to ChunkResponse
        response = ChunkResponse(
            chunk_id=chunk_data["chunk_id"],
            text=chunk_data["text"],
            document_id=chunk_data["document_id"],
            level=chunk_data["level"],
            parent_id=chunk_data.get("parent_id"),
            child_ids=chunk_data.get("child_ids", [])
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunk: {str(e)}")


@router.get("/health")
async def search_health_check():
    """Check health of search components."""
    try:
        retrieval_service = RetrievalService()
        health_status = await retrieval_service.health_check()
        
        return {
            "status": "healthy" if all(health_status.values()) else "unhealthy",
            "components": health_status
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
