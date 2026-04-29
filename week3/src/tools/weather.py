import os
import httpx
from typing import Any, Optional
from .base import BaseTool


class WeatherTool(BaseTool):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="get_weather",
            description="获取指定城市的当前天气信息，支持温度、湿度、天气状况等数据"
        )
        self._api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self._base_url = "https://api.openweathermap.org/data/2.5/weather"
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[int] = None
    
    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称（英文，如 Beijing, Tokyo, London）"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "温度单位，metric为摄氏度，imperial为华氏度，默认为metric",
                    "default": "metric"
                }
            },
            "required": ["city"]
        }
    
    async def execute(self, city: str, units: str = "metric") -> dict[str, Any]:
        if not self._api_key:
            return {
                "success": False,
                "error": "Weather API key not configured. Please set OPENWEATHERMAP_API_KEY environment variable."
            }
        
        if not city or not city.strip():
            return {
                "success": False,
                "error": "City name cannot be empty"
            }
        
        if self._rate_limit_remaining is not None and self._rate_limit_remaining <= 0:
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later."
            }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self._base_url,
                    params={
                        "q": city.strip(),
                        "appid": self._api_key,
                        "units": units
                    }
                )
                
                if "X-RateLimit-Remaining" in response.headers:
                    self._rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
                if "X-RateLimit-Reset" in response.headers:
                    self._rate_limit_reset = int(response.headers["X-RateLimit-Reset"])
                
                if response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"City '{city}' not found. Please check the city name."
                    }
                elif response.status_code == 401:
                    return {
                        "success": False,
                        "error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."
                    }
                elif response.status_code == 429:
                    return {
                        "success": False,
                        "error": "Too many requests. Rate limit exceeded, please try again later."
                    }
                elif response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"API request failed with status code: {response.status_code}"
                    }
                
                data = response.json()
                
                temp_unit = "°C" if units == "metric" else "°F"
                speed_unit = "m/s" if units == "metric" else "mph"
                
                result = {
                    "success": True,
                    "city": data.get("name", city),
                    "country": data.get("sys", {}).get("country", ""),
                    "coordinates": {
                        "latitude": data.get("coord", {}).get("lat"),
                        "longitude": data.get("coord", {}).get("lon")
                    },
                    "weather": {
                        "main": data["weather"][0].get("main", ""),
                        "description": data["weather"][0].get("description", ""),
                        "icon": data["weather"][0].get("icon", "")
                    },
                    "temperature": {
                        "current": f"{data['main']['temp']}{temp_unit}",
                        "feels_like": f"{data['main']['feels_like']}{temp_unit}",
                        "min": f"{data['main']['temp_min']}{temp_unit}",
                        "max": f"{data['main']['temp_max']}{temp_unit}"
                    },
                    "humidity": f"{data['main']['humidity']}%",
                    "pressure": f"{data['main']['pressure']} hPa",
                    "wind": {
                        "speed": f"{data['wind']['speed']} {speed_unit}",
                        "direction": f"{data['wind'].get('deg', 0)}°"
                    },
                    "visibility": f"{data.get('visibility', 0) / 1000} km" if data.get("visibility") else "N/A",
                    "clouds": f"{data.get('clouds', {}).get('all', 0)}%",
                    "sunrise": data.get("sys", {}).get("sunrise"),
                    "sunset": data.get("sys", {}).get("sunset")
                }
                
                return result
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Request timed out. Please try again later."
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "Failed to connect to the weather API. Please check your internet connection."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}"
            }
