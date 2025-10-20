import logging
from typing import Dict, Any, List
from app.services.weather_service import WeatherService
from app.services.location_extractor import LocationExtractor
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class WeatherNode:
    """Weather retrieval node for fetching weather data from OpenWeatherMap API."""
    
    def __init__(self):
        """Initialize weather node."""
        self.weather_service = WeatherService()
        self.location_extractor = LocationExtractor()
        self.llm_service = LLMService()
    
    async def retrieve_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve weather information for the query.
        
        Args:
            state: Current agent state containing query
            
        Returns:
            Updated state with weather context
        """
        try:
            query = state.get("query", "")
            if not query.strip():
                return {
                    **state,
                    "weather_context": "",
                    "weather_sources": []
                }
            
            logger.info(f"Retrieving weather for query: {query}")
            
            # Extract cities from query
            query_cities = self.location_extractor.extract_locations(query)
            
            # Get cities/places extracted from document context
            document_places = state.get("extracted_places", [])
            
            # Combine cities from query and documents
            all_cities = list(set(query_cities + document_places))
            
            if not all_cities:
                logger.warning(f"No cities found in query or documents: {query}")
                return {
                    **state,
                    "weather_context": "No cities mentioned in the query or found in documents.",
                    "weather_sources": []
                }
            
            logger.info(f"Using cities for weather: {all_cities} (from query: {query_cities}, from documents: {document_places})")
            
            # Get weather for all cities
            weather_data = await self.weather_service.get_weather_multiple(all_cities)
            
            # Format weather context
            weather_context = self._format_weather_context(weather_data)
            weather_sources = self._extract_weather_sources(weather_data)
            
            logger.info(f"Retrieved weather for {len(all_cities)} cities")
            
            return {
                **state,
                "weather_context": weather_context,
                "weather_sources": weather_sources
            }
            
        except Exception as e:
            logger.error(f"Weather retrieval failed: {e}")
            return {
                **state,
                "weather_context": f"Weather retrieval error: {str(e)}",
                "weather_sources": []
            }
    
    def _format_weather_context(self, weather_data: List[Dict]) -> str:
        """Format weather data as context for LLM."""
        try:
            if not weather_data:
                return "No weather data available."
            
            context_parts = []
            
            for weather in weather_data:
                if "error" in weather:
                    context_parts.append(f"Weather for {weather.get('city', 'Unknown')}: {weather['error']}")
                else:
                    context_parts.append(self.weather_service.format_weather_summary([weather]))
            
            return "\n\n".join(context_parts) if context_parts else "No weather data available."
            
        except Exception as e:
            logger.error(f"Failed to format weather context: {e}")
            return "Error formatting weather context."
    
    def _extract_weather_sources(self, weather_data: List[Dict]) -> List[Dict]:
        """Extract source information from weather data."""
        try:
            sources = []
            
            for weather in weather_data:
                if "error" not in weather:
                    # Pass the complete weather data structure
                    source = {
                        "type": "weather_api",
                        "city": weather.get("city", ""),
                        "country": weather.get("country", ""),
                        "timestamp": weather.get("timestamp", 0),
                        "temperature": weather.get("temperature", {}),
                        "conditions": weather.get("conditions", {}),
                        "humidity": weather.get("humidity", 0),
                        "pressure": weather.get("pressure", 0),
                        "wind": weather.get("wind", {}),
                        "visibility": weather.get("visibility", 0),
                        "cloudiness": weather.get("cloudiness", 0)
                    }
                    sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to extract weather sources: {e}")
            return []


# Global weather node instance
weather_node = WeatherNode()
