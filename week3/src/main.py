import asyncio
import sys
from typing import Any, Optional

from mcp.server import Server
from mcp.types import (
    Tool as MCPTool,
    TextContent,
    LoggingMessage,
    LoggingLevel,
    Resource as MCPResource,
    Prompt as MCPPrompt,
    PromptMessage,
    Role
)

from .config import config
from .logger import logger
from .capabilities import CapabilityLayer
from .tools import WeatherTool, NewsTool


class MCPServer:
    def __init__(self):
        self._server = Server(
            name=config.mcp_server_name,
            version=config.mcp_server_version
        )
        self._capability_layer = CapabilityLayer()
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self._server.list_tools()
        async def list_tools() -> list[MCPTool]:
            tool_defs = self._capability_layer.list_tools()
            return [
                MCPTool(
                    name=td.name,
                    description=td.description,
                    inputSchema=td.input_schema
                )
                for td in tool_defs
            ]
        
        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            logger.info(f"Calling tool: {name} with arguments: {arguments}")
            
            try:
                result = await self._capability_layer.execute_tool(name, **arguments)
                
                if result.get("success", False):
                    logger.info(f"Tool {name} executed successfully")
                else:
                    logger.warning(f"Tool {name} returned error: {result.get('error')}")
                
                import json
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
                
            except Exception as e:
                logger.error(f"Tool {name} execution failed: {str(e)}")
                import json
                error_result = {
                    "success": False,
                    "error": str(e)
                }
                return [TextContent(type="text", text=json.dumps(error_result, ensure_ascii=False))]
        
        @self._server.list_resources()
        async def list_resources() -> list[MCPResource]:
            resources = self._capability_layer.list_resources()
            return [
                MCPResource(
                    uri=r["uri"],
                    name=r["name"],
                    description=r["description"],
                    mimeType=r["mime_type"]
                )
                for r in resources
            ]
        
        @self._server.read_resource()
        async def read_resource(uri: str) -> TextContent:
            logger.info(f"Reading resource: {uri}")
            
            content = await self._capability_layer.get_resource(uri)
            
            if content is None:
                return TextContent(type="text", text=f"Resource not found: {uri}")
            
            if isinstance(content, (dict, list)):
                import json
                return TextContent(type="text", text=json.dumps(content, ensure_ascii=False, indent=2))
            
            return TextContent(type="text", text=str(content))
        
        @self._server.list_prompts()
        async def list_prompts() -> list[MCPPrompt]:
            prompts = self._capability_layer.list_prompts()
            return [
                MCPPrompt(
                    name=p["name"],
                    description=p["description"]
                )
                for p in prompts
            ]
        
        @self._server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[dict[str, Any]] = None) -> list[PromptMessage]:
            logger.info(f"Getting prompt: {name} with arguments: {arguments}")
            
            args = arguments or {}
            rendered = self._capability_layer.render_prompt(name, **args)
            
            if rendered is None:
                return [
                    PromptMessage(
                        role=Role.USER,
                        content=TextContent(type="text", text=f"Prompt not found: {name}")
                    )
                ]
            
            return [
                PromptMessage(
                    role=Role.USER,
                    content=TextContent(type="text", text=rendered)
                )
            ]
    
    def _register_default_tools(self):
        if config.openweathermap_api_key:
            weather_tool = WeatherTool(api_key=config.openweathermap_api_key)
            self._capability_layer.register_tool(weather_tool)
            logger.info("WeatherTool registered successfully")
        else:
            logger.warning("OPENWEATHERMAP_API_KEY not set, WeatherTool not registered")
        
        if config.newsapi_key:
            news_tool = NewsTool(api_key=config.newsapi_key)
            self._capability_layer.register_tool(news_tool)
            logger.info("NewsTool registered successfully")
        else:
            logger.warning("NEWSAPI_KEY not set, NewsTool not registered")
    
    async def run_stdio(self):
        logger.info(f"Starting MCP server: {config.mcp_server_name} v{config.mcp_server_version}")
        
        init_result = self._capability_layer.initialize(auto_register=False)
        logger.info(f"Capability layer initialized: {init_result}")
        
        self._register_default_tools()
        
        capabilities = self._capability_layer.get_capabilities()
        logger.info(f"Server capabilities: {capabilities}")
        
        from mcp.server.stdio import stdio_server
        
        try:
            await stdio_server(self._server)
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            raise
        finally:
            self._capability_layer.shutdown()
            logger.info("Server shutdown complete")


async def main():
    server = MCPServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
