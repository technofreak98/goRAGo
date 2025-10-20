"""LangGraph-based routing agent for intelligent query processing."""

import time
import logging
from typing import List, Dict, Any, Optional
from app.models import SearchResult, WeatherData, LocationInfo, AgentResponse
from app.agents.rag_graph import rag_graph

logger = logging.getLogger(__name__)


class LangGraphAgent:
    """LangGraph-based routing agent for intelligent query processing."""
    
    def __init__(self):
        """Initialize the agent with the modular RAG graph."""
        self.rag_graph = rag_graph
    
    
    async def process_query(self, query: str) -> AgentResponse:
        """
        Process a user query through the LangGraph agent.
        
        Args:
            query: User's question
            
        Returns:
            AgentResponse with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Use the modular RAG graph
            result = await self.rag_graph.process_query(query)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Build response
            response = self._build_agent_response(result, processing_time)
            
            logger.info(f"Query processed successfully in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._build_error_response(query, str(e), (time.time() - start_time) * 1000)
    
    
    def _build_agent_response(self, result: Dict[str, Any], processing_time: float) -> AgentResponse:
        """Build the final agent response from RAG graph result."""
        try:
            # Extract weather data from sources
            weather_models = []
            location_models = []
            sources = result.get("sources", [])
            
            for source in sources:
                if source.get("type") == "weather_api":
                    # Create properly structured WeatherData using the complete structure from weather service
                    weather_models.append(WeatherData(
                        city=source.get("city", ""),
                        country=source.get("country", ""),
                        temperature=source.get("temperature", {}),
                        conditions=source.get("conditions", {}),
                        humidity=source.get("humidity", 0),
                        pressure=source.get("pressure", 0),
                        wind=source.get("wind", {}),
                        visibility=source.get("visibility", 0),
                        cloudiness=source.get("cloudiness", 0),
                        timestamp=source.get("timestamp", 0)
                    ))
                    location_models.append(LocationInfo(name=source.get("city", "")))
            
            # Get the final answer - try both possible field names
            final_answer = result.get("final_answer", "") or result.get("answer", "")
            
            return AgentResponse(
                answer=final_answer,
                route_taken=result.get("route", "unknown"),
                sources=sources,
                weather_data=weather_models,
                locations=location_models,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error building agent response: {e}")
            import traceback
            traceback.print_exc()
            return self._build_error_response("", str(e), processing_time)
    
    def _build_error_response(self, query: str, error: str, processing_time: float) -> AgentResponse:
        """Build an error response."""
        return AgentResponse(
            answer=f"I apologize, but I encountered an error while processing your query: '{query}'. Please try again or rephrase your question.",
            route_taken="error",
            sources=[],
            weather_data=[],
            locations=[],
            processing_time_ms=processing_time,
            error=error
        )
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all agent components."""
        return {
            "rag_graph": self.rag_graph.health_check(),
            "modular_agents": True
        }
