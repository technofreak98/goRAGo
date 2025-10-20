import logging
from typing import Dict, Any, List, Optional
from app.services.llm_service import LLMService
from app.utils.latency_tracker import track_latency
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)


class GenerationNode:
    """Generation node for creating final responses using retrieved context."""
    
    def __init__(self):
        """Initialize generation node."""
        self.llm_service = LLMService()
    
    @track_latency("generation_response")
    async def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final response using all available context.
        
        Args:
            state: Current agent state with all context
            
        Returns:
            Updated state with final answer
        """
        try:
            query = state.get("query", "")
            document_context = state.get("document_context", "")
            weather_context = state.get("weather_context", "")
            extracted_places = state.get("extracted_places", [])
            route = state.get("route", "")
            
            
            # Combine all available context
            combined_context = self._combine_context(document_context, weather_context, extracted_places)
            
            # Generate response based on available context
            if combined_context.strip():
                final_answer = await self._generate_with_context(query, combined_context, route)
            else:
                final_answer = "I don't have enough information to answer your question. Could you please provide more details or try a different query?"
            
            # Combine all sources
            all_sources = self._combine_sources(state)
            
            return {
                **state,
                "final_answer": final_answer,
                "sources": all_sources
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                **state,
                "final_answer": "I apologize, but I encountered an error while generating a response. Please try again.",
                "sources": []
            }
    
    def _combine_context(self, document_context: str, weather_context: str, extracted_places: Optional[List[str]] = None) -> str:
        """Combine document and weather context with extracted places."""
        try:
            context_parts = []
            
            if document_context.strip():
                context_parts.append("DOCUMENT INFORMATION:")
                context_parts.append(document_context)
            
            if extracted_places:
                context_parts.append(f"\nPLACES MENTIONED IN DOCUMENTS:")
                context_parts.append(f"The following cities/places were mentioned in the documents: {', '.join(extracted_places)}")
            
            if weather_context.strip():
                context_parts.append("\nWEATHER INFORMATION:")
                context_parts.append(weather_context)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to combine context: {e}")
            return ""
    
    async def _generate_with_context(self, query: str, context: str, route: str) -> str:
        """Generate response using LLM with context."""
        try:
            # Create system prompt based on route
            system_prompt = self._get_system_prompt(route)
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = await self.llm_service.chat_completion(messages)
            return response or "I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Failed to generate response with context: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _get_system_prompt(self, route: str) -> str:
        """Get appropriate system prompt based on route."""
        base_prompt = """You are a helpful assistant that answers questions using the provided context information. 
Use the context to answer the user's question accurately and comprehensively.
Always cite your sources when possible and be specific about where information comes from.
If the context doesn't contain enough information, say so politely."""
        
        if route == "weather_only":
            return base_prompt + "\n\nFocus on weather information and provide current, accurate weather data."
        elif route == "document_only":
            return base_prompt + "\n\nFocus on document content and provide detailed information from the documents."
        elif route == "combined":
            return base_prompt + "\n\nCombine weather and document information to provide a comprehensive answer. When cities/places are mentioned in the documents, use that information to provide relevant weather context for those locations."
        else:
            return base_prompt
    
    def _combine_sources(self, state: Dict[str, Any]) -> List[Dict]:
        """Combine all sources from the state."""
        try:
            sources = []
            
            # Add document sources
            document_sources = state.get("document_sources", [])
            if document_sources:
                sources.extend(document_sources)
            
            # Add weather sources
            weather_sources = state.get("weather_sources", [])
            if weather_sources:
                sources.extend(weather_sources)
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to combine sources: {e}")
            return []


# Global generation node instance
generation_node = GenerationNode()
