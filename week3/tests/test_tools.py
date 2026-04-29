import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base import BaseTool, ToolRegistry, ToolDefinition
from src.tools.weather import WeatherTool
from src.tools.news import NewsTool


class MockTool(BaseTool):
    def __init__(self):
        super().__init__(name="mock_tool", description="A mock tool for testing")
    
    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Test parameter"
                }
            },
            "required": ["param1"]
        }
    
    async def execute(self, param1: str) -> dict[str, Any]:
        return {"success": True, "result": param1}


class TestBaseTool:
    def test_initialization(self):
        tool = MockTool()
        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing"
    
    def test_get_definition(self):
        tool = MockTool()
        definition = tool.get_definition()
        
        assert isinstance(definition, ToolDefinition)
        assert definition.name == "mock_tool"
        assert definition.description == "A mock tool for testing"
        assert "properties" in definition.input_schema
    
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = MockTool()
        result = await tool.execute(param1="test_value")
        
        assert result["success"] is True
        assert result["result"] == "test_value"


class TestToolRegistry:
    def setup_method(self):
        ToolRegistry.reset()
        self.registry = ToolRegistry()
    
    def test_singleton(self):
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        assert registry1 is registry2
    
    def test_register_tool(self):
        tool = MockTool()
        self.registry.register(tool)
        
        assert self.registry.get("mock_tool") is tool
        assert "mock_tool" in self.registry.get_all()
    
    def test_register_duplicate_tool(self):
        tool1 = MockTool()
        self.registry.register(tool1)
        
        with pytest.raises(ValueError, match="Tool 'mock_tool' is already registered"):
            tool2 = MockTool()
            self.registry.register(tool2)
    
    def test_unregister_tool(self):
        tool = MockTool()
        self.registry.register(tool)
        
        self.registry.unregister("mock_tool")
        assert self.registry.get("mock_tool") is None
    
    def test_get_all(self):
        tool1 = MockTool()
        self.registry.register(tool1)
        
        all_tools = self.registry.get_all()
        assert len(all_tools) == 1
        assert "mock_tool" in all_tools
    
    def test_list_definitions(self):
        tool = MockTool()
        self.registry.register(tool)
        
        definitions = self.registry.list_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "mock_tool"
    
    def test_clear(self):
        tool = MockTool()
        self.registry.register(tool)
        
        self.registry.clear()
        assert len(self.registry.get_all()) == 0


class TestWeatherTool:
    def setup_method(self):
        self.tool = WeatherTool(api_key="test_api_key")
    
    def test_initialization(self):
        assert self.tool.name == "get_weather"
        assert "天气" in self.tool.description
    
    def test_get_input_schema(self):
        schema = self.tool.get_input_schema()
        
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert "units" in schema["properties"]
        assert "city" in schema["required"]
        assert schema["properties"]["units"]["enum"] == ["metric", "imperial"]
    
    @pytest.mark.asyncio
    async def test_execute_no_api_key(self):
        tool = WeatherTool(api_key=None)
        result = await tool.execute(city="Beijing")
        
        assert result["success"] is False
        assert "API key not configured" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_empty_city(self):
        result = await self.tool.execute(city="")
        
        assert result["success"] is False
        assert "City name cannot be empty" in result["error"]
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_execute_success(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "name": "Beijing",
            "sys": {"country": "CN", "sunrise": 1234567890, "sunset": 1234567890},
            "coord": {"lat": 39.9, "lon": 116.4},
            "weather": [{"main": "Clear", "description": "clear sky", "icon": "01d"}],
            "main": {
                "temp": 25.5,
                "feels_like": 26.0,
                "temp_min": 24.0,
                "temp_max": 27.0,
                "humidity": 50,
                "pressure": 1013
            },
            "wind": {"speed": 2.5, "deg": 180},
            "visibility": 10000,
            "clouds": {"all": 0}
        }
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = await self.tool.execute(city="Beijing")
        
        assert result["success"] is True
        assert result["city"] == "Beijing"
        assert result["country"] == "CN"
        assert "25.5°C" in result["temperature"]["current"]
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_execute_404_error(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = await self.tool.execute(city="NonexistentCity")
        
        assert result["success"] is False
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_execute_401_error(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = await self.tool.execute(city="Beijing")
        
        assert result["success"] is False
        assert "Invalid API key" in result["error"]


class TestNewsTool:
    def setup_method(self):
        self.tool = NewsTool(api_key="test_api_key")
    
    def test_initialization(self):
        assert self.tool.name == "search_news"
        assert "新闻" in self.tool.description
    
    def test_get_input_schema(self):
        schema = self.tool.get_input_schema()
        
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "category" in schema["properties"]
        assert "language" in schema["properties"]
        assert "sort_by" in schema["properties"]
        assert "page_size" in schema["properties"]
        assert "page" in schema["properties"]
        assert len(schema.get("required", [])) == 0
    
    @pytest.mark.asyncio
    async def test_execute_no_api_key(self):
        tool = NewsTool(api_key=None)
        result = await tool.execute(query="technology")
        
        assert result["success"] is False
        assert "API key not configured" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_no_query_or_category(self):
        result = await self.tool.execute()
        
        assert result["success"] is False
        assert "query' or 'category'" in result["error"]
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_execute_success(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "Test News 1",
                    "description": "Test description 1",
                    "content": "Test content 1",
                    "url": "https://example.com/news1",
                    "urlToImage": "https://example.com/image1.jpg",
                    "source": {"id": "test-source", "name": "Test Source"},
                    "author": "Test Author",
                    "publishedAt": "2024-01-01T12:00:00Z"
                },
                {
                    "title": "Test News 2",
                    "description": "Test description 2",
                    "content": "Test content 2",
                    "url": "https://example.com/news2",
                    "urlToImage": "https://example.com/image2.jpg",
                    "source": {"id": None, "name": "Another Source"},
                    "author": None,
                    "publishedAt": "2024-01-01T11:00:00Z"
                }
            ]
        }
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = await self.tool.execute(query="technology", language="en")
        
        assert result["success"] is True
        assert result["total_results"] == 2
        assert result["articles_count"] == 2
        assert len(result["articles"]) == 2
        assert result["articles"][0]["title"] == "Test News 1"
        assert result["articles"][0]["source"]["name"] == "Test Source"
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_execute_with_category(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "title": "Business News",
                    "description": "Business description",
                    "url": "https://example.com/business",
                    "source": {"id": "business", "name": "Business News"},
                    "publishedAt": "2024-01-01T10:00:00Z"
                }
            ]
        }
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = await self.tool.execute(category="business", language="en")
        
        assert result["success"] is True
        assert result["total_results"] == 1
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_execute_401_error(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {"message": "Invalid API key"}
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = await self.tool.execute(query="technology")
        
        assert result["success"] is False
        assert "Invalid API key" in result["error"]
