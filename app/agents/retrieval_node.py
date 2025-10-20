import logging
from typing import Dict, Any, List, Optional
from app.services.retrieval_service import RetrievalService
from app.services.query_preprocessor import QueryPreprocessor
from app.services.location_extractor import LocationExtractor
from app.services.llm_service import LLMService
from app.models import SearchQuery
from app.config import settings
from app.utils.latency_tracker import track_latency
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)


class RetrievalNode:
    """Document retrieval node for fetching relevant documents from Elasticsearch."""
    
    def __init__(self):
        """Initialize retrieval node."""
        self.retrieval_service = RetrievalService()
        self.query_preprocessor = QueryPreprocessor()
        self.location_extractor = LocationExtractor()
        self.llm_service = LLMService()
        self.top_k = settings.final_top_k
    
    @track_latency("retrieval_documents")
    async def retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            state: Current agent state containing query
            
        Returns:
            Updated state with document context
        """
        try:
            query = state.get("query", "")
            if not query.strip():
                return {
                    **state,
                    "document_context": "",
                    "document_sources": [],
                    "extracted_places": []
                }
            
            
            # Preprocess query for better retrieval
            processed_query, keywords = self.query_preprocessor.preprocess_query(query)
            
            # Create search query
            # Note: top_k here is just a hint, the retrieval service will use initial_top_k (20) 
            # for initial search and final_top_k (10) for final results
            search_query = SearchQuery(
                query=processed_query,
                top_k=self.top_k,  # This will be overridden by initial_top_k in retrieval service
                rerank=True,
                compression=True
            )
            
            # Retrieve documents
            search_response = await self.retrieval_service.search(search_query)
            retrieved_docs = search_response.results
            
            if not retrieved_docs:
                logger.warning(f"No documents found for query: {query}")
                return {
                    **state,
                    "document_context": "No relevant documents found.",
                    "document_sources": [],
                    "extracted_places": []
                }
            
            # Format context with parent information
            document_context = self._format_document_context(retrieved_docs, search_response.combined_context)
            document_sources = self._extract_sources(retrieved_docs)
            
            # Extract cities/places from document context only if query is weather-related
            route = state.get("route", "")
            if route in ["combined", "weather_only"]:
                extracted_places = await self._extract_places_from_context(document_context)
            else:
                extracted_places = []
            
            return {
                **state,
                "document_context": document_context,
                "document_sources": document_sources,
                "extracted_places": extracted_places
            }
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return {
                **state,
                "document_context": f"Document retrieval error: {str(e)}",
                "document_sources": [],
                "extracted_places": []
            }
    
    def _format_document_context(self, results: List, combined_context: Optional[str] = None) -> str:
        """Format retrieved documents as context for LLM."""
        try:
            if combined_context:
                return combined_context
            
            if not results:
                return "No relevant documents found."
            
            context_parts = []
            
            for i, result in enumerate(results, 1):
                chunk_info = f"Document {i} (Relevance: {result.relevance_score:.3f}):\n"
                chunk_info += f"Source: {result.document_id}\n"
                
                if result.section_info:
                    chunk_info += f"Section: {result.section_info.title}\n"
                if result.chapter_info:
                    chunk_info += f"Chapter: {result.chapter_info.title}\n"
                if result.part_info:
                    chunk_info += f"Part: {result.part_info.title}\n"
                
                chunk_info += f"Content: {result.text}\n"
                context_parts.append(chunk_info)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to format document context: {e}")
            return "Error formatting document context."
    
    def _extract_sources(self, results: List) -> List[Dict]:
        """Extract source information from results."""
        try:
            sources = []
            
            for result in results:
                source = {
                    "type": "document",
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "relevance": result.relevance_score or 0.0,
                    "text": result.text[:200] + "..." if len(result.text) > 200 else result.text
                }
                
                if result.section_info:
                    source["section"] = result.section_info.title
                if result.chapter_info:
                    source["chapter"] = result.chapter_info.title
                if result.part_info:
                    source["part"] = result.part_info.title
                
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to extract sources: {e}")
            return []
    
    async def _extract_places_from_context(self, document_context: str) -> List[str]:
        """Extract cities/places from document context using LLM."""
        try:
            if not document_context.strip():
                logger.info("No document context to extract places from")
                return []
            
            logger.info(f"Extracting places from document context (length: {len(document_context)})")
            
            # If context is too long, chunk it to avoid token limits
            max_chunk_size = 2000  # Conservative chunk size to stay within token limits
            all_places = []
            
            if len(document_context) > max_chunk_size:
                logger.info(f"Context too long ({len(document_context)} chars), chunking for place extraction")
                
                # Split context into chunks
                chunks = []
                start = 0
                while start < len(document_context):
                    end = start + max_chunk_size
                    # Try to break at a sentence or paragraph boundary
                    if end < len(document_context):
                        # Look for sentence endings within the last 200 chars
                        break_point = end
                        for i in range(min(200, end - start)):
                            if document_context[end - i] in '.!?':
                                break_point = end - i + 1
                                break
                        end = break_point
                    
                    chunk = document_context[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = end
                
                logger.info(f"Split context into {len(chunks)} chunks for place extraction")
                
                # Extract places from each chunk
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)})")
                    chunk_places = await self.llm_service.extract_places_from_text(chunk)
                    if chunk_places:
                        all_places.extend(chunk_places)
                        logger.debug(f"Chunk {i+1} places: {chunk_places}")
            else:
                # Use LLM to extract places from the document context
                all_places = await self.llm_service.extract_places_from_text(document_context)
            
            logger.info(f"LLM extracted places: {all_places}")
            
            # Remove duplicates and filter out empty strings
            unique_places = list(set([place.strip() for place in all_places if place.strip()]))
            logger.info(f"Final unique places: {unique_places}")
            
            return unique_places
            
        except Exception as e:
            logger.error(f"Failed to extract places from context: {e}")
            return []


# Global retrieval node instance
retrieval_node = RetrievalNode()
