"""Weather service for fetching data from OpenWeatherMap API."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for fetching weather data from OpenWeatherMap API."""
    
    def __init__(self):
        """Initialize weather service with API configuration."""
        self.api_key = settings.openweather_api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.timeout = 10.0
        
        if not self.api_key or self.api_key == "your_openweather_api_key_here":
            logger.warning("OpenWeatherMap API key not configured")
    
    async def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        """
        Get current weather for a single city.
        
        Args:
            city: City name (e.g., "Rome", "Venice, Italy")
            
        Returns:
            Weather data dictionary or None if error
        """
        try:
            if not self.api_key:
                logger.error("OpenWeatherMap API key not configured")
                return None
            
            # Clean city name
            clean_city = self._clean_city_name(city)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/weather"
                params = {
                    "q": clean_city,
                    "appid": self.api_key,
                    "units": "metric"  # Celsius
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                weather_info = self._parse_weather_data(data, clean_city)
                
                logger.info(f"Weather data fetched for {clean_city}")
                return weather_info
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"City not found: {city}")
            else:
                logger.error(f"HTTP error fetching weather for {city}: {e}")
            return None
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching weather for {city}")
            return None
        except Exception as e:
            logger.error(f"Error fetching weather for {city}: {e}")
            return None
    
    async def get_weather_multiple(self, cities: List[str]) -> List[Dict[str, Any]]:
        """
        Get weather data for multiple cities concurrently.
        
        Args:
            cities: List of city names
            
        Returns:
            List of weather data dictionaries
        """
        try:
            if not cities:
                return []
            
            # Create tasks for concurrent requests
            tasks = [self.get_weather(city) for city in cities]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            weather_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching weather for {cities[i]}: {result}")
                elif result is not None:
                    weather_data.append(result)
            
            logger.info(f"Weather data fetched for {len(weather_data)}/{len(cities)} cities")
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather for multiple cities: {e}")
            return []
    
    def _clean_city_name(self, city: str) -> str:
        """Clean and normalize city name for API call."""
        # Remove extra whitespace
        city = city.strip()
        
        # Handle common city name variations
        city_mappings = {
            "rome": "Rome, IT",
            "venice": "Venice, IT",
            "florence": "Florence, IT",
            "milan": "Milan, IT",
            "naples": "Naples, IT",
            "turin": "Turin, IT",
            "bologna": "Bologna, IT",
            "genoa": "Genoa, IT",
            "paris": "Paris, FR",
            "london": "London, GB",
            "madrid": "Madrid, ES",
            "berlin": "Berlin, DE"
        }
        
        city_lower = city.lower()
        if city_lower in city_mappings:
            return city_mappings[city_lower]
        
        # If no mapping found, return as-is
        return city
    
    def _parse_weather_data(self, data: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Parse OpenWeatherMap API response into standardized format."""
        try:
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            wind = data.get("wind", {})
            sys = data.get("sys", {})
            
            weather_info = {
                "city": city,
                "country": sys.get("country", "Unknown"),
                "temperature": {
                    "current": round(main.get("temp", 0), 1),
                    "feels_like": round(main.get("feels_like", 0), 1),
                    "min": round(main.get("temp_min", 0), 1),
                    "max": round(main.get("temp_max", 0), 1)
                },
                "conditions": {
                    "main": weather.get("main", "Unknown"),
                    "description": weather.get("description", "Unknown"),
                    "icon": weather.get("icon", "")
                },
                "humidity": main.get("humidity", 0),
                "pressure": main.get("pressure", 0),
                "wind": {
                    "speed": wind.get("speed", 0),
                    "direction": wind.get("deg", 0)
                },
                "visibility": data.get("visibility", 0) / 1000,  # Convert to km
                "cloudiness": data.get("clouds", {}).get("all", 0),
                "timestamp": data.get("dt", 0)
            }
            
            return weather_info
            
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return {
                "city": city,
                "error": "Failed to parse weather data"
            }
    
    def format_weather_summary(self, weather_data: List[Dict[str, Any]]) -> str:
        """
        Format weather data into a human-readable summary.
        
        Args:
            weather_data: List of weather data dictionaries
            
        Returns:
            Formatted weather summary string
        """
        if not weather_data:
            return "No weather data available."
        
        summary_parts = []
        
        for weather in weather_data:
            if "error" in weather:
                summary_parts.append(f"❌ {weather['city']}: {weather['error']}")
                continue
            
            city = weather.get("city", "Unknown")
            temp = weather["temperature"]["current"]
            description = weather["conditions"]["description"].title()
            humidity = weather["humidity"]
            
            # Add weather emoji based on conditions
            emoji = self._get_weather_emoji(weather["conditions"]["main"])
            
            summary_parts.append(
                f"{emoji} **{city}**: {temp}°C, {description}, "
                f"Humidity: {humidity}%"
            )
        
        return "\n".join(summary_parts)
    
    def _get_weather_emoji(self, condition: str) -> str:
        """Get weather emoji based on condition."""
        emoji_map = {
            "Clear": "☀️",
            "Clouds": "☁️",
            "Rain": "🌧️",
            "Drizzle": "🌦️",
            "Thunderstorm": "⛈️",
            "Snow": "❄️",
            "Mist": "🌫️",
            "Fog": "🌫️",
            "Haze": "🌫️",
            "Dust": "🌪️",
            "Sand": "🌪️",
            "Ash": "🌋",
            "Squall": "💨",
            "Tornado": "🌪️"
        }
        
        return emoji_map.get(condition, "🌤️")
    
    async def health_check(self) -> bool:
        """Check if weather service is working."""
        try:
            # Try to fetch weather for a known city
            test_result = await self.get_weather("London")
            return test_result is not None and "error" not in test_result
        except Exception as e:
            logger.error(f"Weather service health check failed: {e}")
            return False
