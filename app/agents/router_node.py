import logging
from typing import Dict, Any
from app.services.llm_service import LLMService
from app.config import settings
from app.utils.latency_tracker import track_latency
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)


class RouterNode:
    """Router node for classifying queries and determining routing decisions."""
    
    def __init__(self):
        """Initialize router node."""
        self.llm_service = LLMService()
        self.categories = ["weather_only", "document_only", "combined", "out_of_scope"]
    
    @track_latency("router_classify_query")
    async def route_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a query based on its content and intent.
        
        Args:
            state: Current agent state containing query and other information
            
        Returns:
            Updated state with routing decision
        """
        try:
            query = state.get("query", "")
            if not query.strip():
                return {
                    **state,
                    "route": "out_of_scope",
                    "confidence": 0.0,
                    "reasoning": "Empty query"
                }
            
            # Use LLM to classify the query
            classification = await self.llm_service.classify_query_intent(query)
            
            # Map the classification to our categories
            route = self._map_classification_to_route(classification)
            confidence = 0.8  # Default confidence for now
            reasoning = f"Query classified as: {classification}"
            
            # Validate route
            if route not in self.categories:
                route = "out_of_scope"
                confidence = 0.0
                reasoning = "Invalid classification result"
            
            return {
                **state,
                "route": route,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Router node failed: {e}")
            return {
                **state,
                "route": "out_of_scope",
                "confidence": 0.0,
                "reasoning": f"Router error: {str(e)}"
            }
    
    def _map_classification_to_route(self, classification: str) -> str:
        """Map LLM classification to our route categories."""
        classification_lower = classification.lower().strip()
        
        # Direct mapping of classification results
        if classification_lower == "weather":
            return "weather_only"
        elif classification_lower == "document":
            return "document_only"
        elif classification_lower == "combined":
            return "combined"
        elif classification_lower == "guardrails":
            return "out_of_scope"
        else:
            # Fallback to keyword matching for robustness
            if "weather" in classification_lower and "document" not in classification_lower:
                return "weather_only"
            elif "document" in classification_lower and "weather" not in classification_lower:
                return "document_only"
            elif "weather" in classification_lower and "document" in classification_lower:
                return "combined"
            else:
                return "out_of_scope"
    
    def should_continue_to_weather(self, state: Dict[str, Any]) -> str:
        """Check if should continue to weather node."""
        route = state.get("route", "")
        return "weather" if route in ["weather_only", "combined"] else "__end__"
    
    def should_continue_to_documents(self, state: Dict[str, Any]) -> str:
        """Check if should continue to document retrieval node."""
        route = state.get("route", "")
        return "documents" if route in ["document_only", "combined"] else "__end__"
    
    def should_continue_to_guardrail(self, state: Dict[str, Any]) -> str:
        """Check if should continue to guardrail node."""
        route = state.get("route", "")
        return "guardrail" if route == "out_of_scope" else "__end__"


# Global router node instance
router_node = RouterNode()
