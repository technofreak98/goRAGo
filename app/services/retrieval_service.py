"""Retrieval service for hybrid search with reranking and contextual compression."""

import time
import logging
from typing import List, Dict, Any, Optional
from app.models import SearchQuery, SearchResult, SearchResponse
from app.services.elasticsearch_service import ElasticsearchService
from app.services.embedding_service import EmbeddingService
from app.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for hybrid retrieval with reranking and compression."""
    
    def __init__(self):
        """Initialize retrieval service."""
        self.es_service = ElasticsearchService()
        self.embedding_service = EmbeddingService()
        
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Perform hybrid search with optional reranking and compression."""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query.query)
            
            # Prepare filters
            filters = {}
            if query.filter_by_part:
                filters["part_number"] = query.filter_by_part
            if query.filter_by_chapter:
                filters["chapter_number"] = query.filter_by_chapter
            
            # Perform hybrid search with initial_top_k to get more candidates
            initial_results = await self.es_service.hybrid_search(
                query=query.query,
                query_embedding=query_embedding,
                top_k=settings.initial_top_k,  # Use initial_top_k (20) instead of query.top_k
                filters=filters
            )
            
            # Apply reranking if requested
            if query.rerank and len(initial_results) > settings.final_top_k:
                reranked_results = await self._rerank_results(query.query, initial_results)
                final_results = reranked_results[:settings.final_top_k]  # Get top 10 after reranking
                used_reranking = True
            else:
                final_results = initial_results[:settings.final_top_k]  # Get top 10 without reranking
                used_reranking = False
            
            # # Apply contextual compression if requested
            # if query.compression:
            #     compressed_results = await self._compress_results(query.query, final_results)
            #     final_results = compressed_results
            #     used_compression = True
            # else:
            used_compression = False
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Calculate RAG-specific metrics
            total_tokens = sum(result.token_count or 0 for result in final_results)
            relevance_scores = [result.relevance_score for result in final_results if result.relevance_score is not None]
            max_relevance = max(relevance_scores) if relevance_scores else 0.0
            min_relevance = min(relevance_scores) if relevance_scores else 0.0
            
            # Create combined context for RAG
            combined_context = self._create_combined_context(final_results)
            
            # Create search response
            response = SearchResponse(
                query=query.query,
                results=final_results,
                total_results=len(final_results),
                processing_time_ms=processing_time,
                used_reranking=used_reranking,
                used_compression=used_compression,
                # RAG-specific fields
                combined_context=combined_context,
                total_tokens=total_tokens,
                max_relevance_score=max_relevance,
                min_relevance_score=min_relevance
            )
            
            logger.info(f"Search completed: {len(initial_results)} initial → {len(final_results)} final results in {processing_time:.2f}ms (reranking: {used_reranking})")
            return response
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using cross-encoder or lightweight reranker."""
        try:
            # For now, implement a simple reranking based on query-term overlap
            # In production, you would use a cross-encoder model or Elasticsearch's built-in reranking
            
            reranked_results = []
            
            for result in results:
                # Calculate rerank score based on query term overlap
                rerank_score = self._calculate_rerank_score(query, result.text)
                
                # Combine original score with rerank score
                combined_score = (result.score * 0.7) + (rerank_score * 0.3)
                
                # Create new result with updated score
                reranked_result = SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=combined_score,
                    document_id=result.document_id,
                    level=result.level,
                    parent_id=result.parent_id,
                    child_ids=result.child_ids,
                    rank=result.rank,
                    parent_window=result.parent_window,
                    section_info=result.section_info,
                    chapter_info=result.chapter_info,
                    part_info=result.part_info,
                    # RAG-specific fields
                    context=result.context,
                    relevance_score=min(combined_score / 100.0, 1.0),  # Normalize to 0-1
                    token_count=result.token_count
                )
                reranked_results.append(reranked_result)
            
            # Sort by combined score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            logger.info(f"Reranked {len(results)} results → {len(reranked_results)} final results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results  # Return original results if reranking fails
    
    async def _compress_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Apply contextual compression to reduce token count."""
        try:
            compressed_results = []
            total_tokens = 0
            
            for result in results:
                # Estimate tokens in current result
                current_tokens = self._estimate_tokens(result.text)
                
                # Check if adding this result would exceed compression target
                if total_tokens + current_tokens > settings.max_tokens_compression:
                    # Try to compress this result to fit
                    compressed_text = self._compress_text(query, result.text)
                    compressed_tokens = self._estimate_tokens(compressed_text)
                    
                    if total_tokens + compressed_tokens <= settings.max_tokens_compression:
                        # Use compressed version
                        result.text = compressed_text
                        compressed_results.append(result)
                        total_tokens += compressed_tokens
                    else:
                        # Skip this result to stay within token limit
                        break
                else:
                    compressed_results.append(result)
                    total_tokens += current_tokens
            
            logger.info(f"Compressed results to {len(compressed_results)} items, ~{total_tokens} tokens")
            return compressed_results
            
        except Exception as e:
            logger.error(f"Error in compression: {e}")
            return results  # Return original results if compression fails
    
    def _calculate_rerank_score(self, query: str, text: str) -> float:
        """Calculate rerank score based on query-term overlap."""
        try:
            # Simple TF-IDF-like scoring
            query_terms = set(query.lower().split())
            text_terms = set(text.lower().split())
            
            # Calculate overlap
            overlap = len(query_terms.intersection(text_terms))
            
            # Normalize by query length
            if len(query_terms) == 0:
                return 0.0
            
            score = overlap / len(query_terms)
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def _compress_text(self, query: str, text: str) -> str:
        """Compress text by extracting most relevant parts."""
        try:
            # Split text into sentences
            sentences = text.split('. ')
            
            # Score sentences based on query term overlap
            scored_sentences = []
            for sentence in sentences:
                score = self._calculate_rerank_score(query, sentence)
                scored_sentences.append((sentence, score))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Take sentences until we reach a reasonable length
            compressed_sentences = []
            total_length = 0
            max_length = len(text) // 2  # Compress to at most half the original length
            
            for sentence, score in scored_sentences:
                if total_length + len(sentence) <= max_length:
                    compressed_sentences.append(sentence)
                    total_length += len(sentence)
                else:
                    break
            
            # Join sentences back
            compressed_text = '. '.join(compressed_sentences)
            
            # If compression resulted in very short text, take first part of original
            if len(compressed_text) < len(text) // 4:
                compressed_text = text[:max_length] + "..."
            
            return compressed_text
            
        except Exception:
            return text  # Return original text if compression fails
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            # Simple estimation: ~4 characters per token for English
            return len(text) // 4
        except Exception:
            return 0
    
    def _create_combined_context(self, results: List[SearchResult]) -> str:
        """Create combined context from search results for RAG."""
        try:
            if not results:
                return ""
            
            # Sort by relevance score (highest first)
            sorted_results = sorted(results, key=lambda x: x.relevance_score or 0, reverse=True)
            
            # Create context sections
            context_sections = []
            for i, result in enumerate(sorted_results, 1):
                # Add section header with relevance info
                relevance_info = f" (Relevance: {result.relevance_score:.2f})" if result.relevance_score else ""
                section_header = f"--- Source {i}{relevance_info} ---"
                
                # Add the context (text + parent_window if available)
                context_text = result.context or result.text
                
                # Add section info if available
                section_info = ""
                if result.section_info:
                    section_info = f"\n[Section: {result.section_info.title}]"
                elif result.chapter_info:
                    section_info = f"\n[Chapter: {result.chapter_info.title}]"
                elif result.part_info:
                    section_info = f"\n[Part: {result.part_info.title}]"
                
                context_sections.append(f"{section_header}{section_info}\n{context_text}")
            
            return "\n\n".join(context_sections)
            
        except Exception as e:
            logger.error(f"Error creating combined context: {e}")
            # Fallback to simple concatenation
            return "\n\n".join([result.text for result in results])
    
    async def get_chunk_with_context(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk with its full parent context."""
        try:
            chunk_data = await self.es_service.get_chunk(chunk_id)
            return chunk_data
        except Exception as e:
            logger.error(f"Error getting chunk with context: {e}")
            return None
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of retrieval service components."""
        return {
            "elasticsearch": await self._check_elasticsearch_health(),
            "embeddings": await self.embedding_service.health_check()
        }
    
    async def _check_elasticsearch_health(self) -> bool:
        """Check Elasticsearch health."""
        try:
            # Try to get cluster health
            response = self.es_service.client.cluster.health()
            return response.get("status") in ["green", "yellow"]
        except Exception:
            return False
