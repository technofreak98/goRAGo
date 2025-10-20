"""LLM service for generating answers using OpenAI GPT models."""

import logging
from typing import List, Dict, Any, Optional
import openai
from app.config import settings
from app.utils.cost_tracker import cost_tracker
from app.utils.latency_tracker import track_latency
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)


class LLMService:
    """Service for generating answers using OpenAI GPT models."""
    
    def __init__(self):
        """Initialize LLM service with OpenAI client."""
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = 2000
    
    @track_latency("llm_generate_answer")
    async def generate_answer(self, query: str, context: str, weather_data: Optional[List[Dict]] = None) -> str:
        """
        Generate an answer using GPT model with context and optional weather data.
        
        Args:
            query: User's question
            context: Retrieved document context
            weather_data: Optional weather data for combined queries
            
        Returns:
            Generated answer string
        """
        try:
            # Build the prompt based on available data
            prompt = self._build_prompt(query, context, weather_data)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._get_fallback_response(query, weather_data)
    
    @track_latency("llm_classify_query")
    async def classify_query_intent(self, query: str) -> str:
        """
        Classify query intent to determine routing.
        
        Args:
            query: User's question
            
        Returns:
            Intent classification: "document", "weather", "combined", "guardrails"
        """
        try:
            prompt = self._build_classification_prompt(query)
            
            response = await self._generate_response(prompt, max_tokens=50)
            
            # Parse response to get classification
            classification = self._parse_classification(response)
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return "guardrails"  # Default to guardrails on error
    
    def _build_prompt(self, query: str, context: str, weather_data: Optional[List[Dict]] = None) -> str:
        """Build the prompt for answer generation."""
        
        # Base system prompt
        system_prompt = """You are a helpful travel and literature assistant. You have access to document content and weather information to help users plan their travels based on literary works.

Your role:
- Answer questions about places mentioned in literature
- Provide weather information for travel planning
- Combine literary context with current weather data when relevant
- Be helpful, accurate, and engaging

Guidelines:
- Use the provided document context to answer questions about places, people, and events
- When weather data is provided, incorporate it naturally into your response
- If you don't have enough information, say so clearly
- Focus on travel-relevant information
- Be conversational but informative"""

        # Build context section
        context_section = f"""
DOCUMENT CONTEXT:
{context}
"""

        # Build weather section if available
        weather_section = ""
        if weather_data:
            weather_section = f"""
CURRENT WEATHER INFORMATION:
{self._format_weather_for_prompt(weather_data)}
"""

        # Build the complete prompt
        prompt = f"""{system_prompt}

{context_section}{weather_section}

USER QUESTION: {query}

Please provide a helpful and informative answer based on the available information. If weather data is provided, incorporate it naturally into your response about travel planning."""

        return prompt
    
    def _build_classification_prompt(self, query: str) -> str:
        """Build prompt for query intent classification."""
        
        return f"""Classify the following query into one of these categories:

1. "document" - Questions about literature, authors, places in books, historical information
   Examples: "What places did Mark Twain visit?", "Tell me about Rome in literature"

2. "weather" - Questions about current weather conditions
   Examples: "What's the weather in Rome?", "Is it raining in Venice?"

3. "combined" - Questions that combine literature/travel with weather
   Examples: "I want to visit places Twain went to in Italy - what's the weather?", "Planning a trip to places mentioned in the book, what's the weather like?"

4. "guardrails" - Questions outside the scope of travel and literature
   Examples: "Explain quantum physics", "How to cook pasta", "What's the stock market doing?"

Query: "{query}"

Respond with only one word: document, weather, combined, or guardrails"""
    
    async def _generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                timeout=30
            )
            
            # Track cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost_metrics = cost_tracker.calculate_cost(self.model, prompt_tokens, completion_tokens)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    # @track_latency("llm_chat_completion")
    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using chat completion API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated response string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30
            )
            
            # Track cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost_metrics = cost_tracker.calculate_cost(self.model, prompt_tokens, completion_tokens)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling OpenAI chat completion API: {e}")
            raise
    
    def _parse_classification(self, response: str) -> str:
        """Parse classification response."""
        response_lower = response.lower().strip()
        
        if "document" in response_lower:
            return "document"
        elif "weather" in response_lower:
            return "weather"
        elif "combined" in response_lower:
            return "combined"
        else:
            return "guardrails"
    
    def _format_weather_for_prompt(self, weather_data: List[Dict]) -> str:
        """Format weather data for inclusion in prompt."""
        if not weather_data:
            return "No weather data available."
        
        weather_lines = []
        for weather in weather_data:
            if "error" in weather:
                weather_lines.append(f"❌ {weather['city']}: {weather['error']}")
                continue
            
            city = weather.get("city", "Unknown")
            temp = weather["temperature"]["current"]
            description = weather["conditions"]["description"]
            humidity = weather["humidity"]
            
            weather_lines.append(
                f"🌤️ {city}: {temp}°C, {description}, Humidity: {humidity}%"
            )
        
        return "\n".join(weather_lines)
    
    def _get_fallback_response(self, query: str, weather_data: Optional[List[Dict]] = None) -> str:
        """Get fallback response when LLM fails."""
        if weather_data:
            weather_summary = self._format_weather_for_prompt(weather_data)
            return f"""I apologize, but I'm having trouble processing your request right now. 

However, I can provide you with the current weather information:

{weather_summary}

Please try rephrasing your question, and I'll do my best to help you with your travel planning based on literature."""
        else:
            return """I apologize, but I'm having trouble processing your request right now. 

Please try rephrasing your question, and I'll do my best to help you with information about places mentioned in literature and travel planning."""
    
    async def health_check(self) -> bool:
        """Check if LLM service is working."""
        try:
            test_response = await self._generate_response("Hello", max_tokens=10)
            return bool(test_response)
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return False
    
    @track_latency("llm_extract_places")
    async def extract_places_from_text(self, text: str) -> List[str]:
        """
        Extract city/place names from text using LLM.
        
        Args:
            text: Text to extract places from
            
        Returns:
            List of place names found in the text
        """
        try:
            system_prompt = """You are a helpful assistant that extracts city and place names from text.
            Return only the names of cities, towns, countries, or geographical locations mentioned in the text.
            Return the results as a simple list, one place per line.
            Do not include explanations or additional text.
            If no places are found, return an empty list."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all city and place names from this text:\n\n{text}"}
            ]
            
            response = await self.chat_completion(messages)
            
            if not response:
                return []
            
            # Parse the response into a list
            places = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Filter out common false positives
            filtered_places = []
            for place in places:
                place = place.strip()
                if place and len(place) > 1 and not place.lower() in ['city', 'town', 'place', 'location', 'country']:
                    filtered_places.append(place)
            
            return filtered_places
            
        except Exception as e:
            logger.error(f"Error extracting places from text: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
