from typing import Any, Optional
from .tool_manager import ToolManager
from .resource_manager import ResourceManager
from .prompt_manager import PromptManager
from ..tools.base import BaseTool, ToolDefinition


class CapabilityLayer:
    def __init__(
        self,
        tool_manager: Optional[ToolManager] = None,
        resource_manager: Optional[ResourceManager] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        self._tool_manager = tool_manager or ToolManager()
        self._resource_manager = resource_manager or ResourceManager()
        self._prompt_manager = prompt_manager or PromptManager()
        self._initialized = False
    
    @property
    def tool_manager(self) -> ToolManager:
        return self._tool_manager
    
    @property
    def resource_manager(self) -> ResourceManager:
        return self._resource_manager
    
    @property
    def prompt_manager(self) -> PromptManager:
        return self._prompt_manager
    
    def initialize(self, auto_register: bool = True) -> dict[str, Any]:
        if self._initialized:
            return {
                "success": True,
                "message": "Capability layer already initialized",
                "tools_count": len(self._tool_manager.list_tools()),
                "resources_count": len(self._resource_manager.list_resources()),
                "prompts_count": len(self._prompt_manager.list_prompts())
            }
        
        tools_registered = 0
        prompts_registered = 0
        
        if auto_register:
            tools_registered = self._tool_manager.auto_register_tools()
            prompts_registered = self._prompt_manager.auto_register_prompts()
        
        self._initialized = True
        
        return {
            "success": True,
            "message": "Capability layer initialized successfully",
            "tools_registered": tools_registered,
            "prompts_registered": prompts_registered
        }
    
    def get_capabilities(self) -> dict[str, Any]:
        return {
            "tools": {
                "enabled": True,
                "count": len(self._tool_manager.list_tools()),
                "names": self._tool_manager.list_tool_names()
            },
            "resources": {
                "enabled": True,
                "count": len(self._resource_manager.list_resources())
            },
            "prompts": {
                "enabled": True,
                "count": len(self._prompt_manager.list_prompts())
            }
        }
    
    def list_tools(self) -> list[ToolDefinition]:
        return self._tool_manager.list_tools()
    
    async def execute_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        return await self._tool_manager.execute_tool(tool_name, **kwargs)
    
    def register_tool(self, tool: BaseTool) -> None:
        self._tool_manager.register_tool(tool)
    
    def list_resources(self) -> list[dict[str, Any]]:
        return self._resource_manager.list_resources()
    
    async def get_resource(self, uri: str) -> Optional[Any]:
        return await self._resource_manager.get_resource_content(uri)
    
    def list_prompts(self) -> list[dict[str, Any]]:
        return self._prompt_manager.list_prompts()
    
    def render_prompt(self, name: str, **kwargs) -> Optional[str]:
        return self._prompt_manager.render_prompt(name, **kwargs)
    
    def shutdown(self) -> None:
        self._tool_manager.clear()
        self._resource_manager.clear()
        self._prompt_manager.clear()
        self._initialized = False
