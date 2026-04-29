import os
import importlib
import inspect
from pathlib import Path
from typing import Any, Optional, Type
from ..tools.base import BaseTool, ToolRegistry, ToolDefinition


class ToolManager:
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or ToolRegistry()
    
    @property
    def registry(self) -> ToolRegistry:
        return self._registry
    
    def register_tool(self, tool: BaseTool) -> None:
        self._registry.register(tool)
    
    def unregister_tool(self, tool_name: str) -> None:
        self._registry.unregister(tool_name)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self._registry.get(tool_name)
    
    def list_tools(self) -> list[ToolDefinition]:
        return self._registry.list_definitions()
    
    def list_tool_names(self) -> list[str]:
        return list(self._registry.get_all().keys())
    
    async def execute_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        tool = self._registry.get(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        return await tool.execute(**kwargs)
    
    def auto_register_tools(self, tools_dir: Optional[Path] = None) -> int:
        if tools_dir is None:
            tools_dir = Path(__file__).parent.parent / "tools"
        
        registered_count = 0
        
        for file_path in tools_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            module_name = file_path.stem
            module_path = f"{__package__.rsplit('.', 1)[0]}.tools.{module_name}"
            
            try:
                module = importlib.import_module(module_path)
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseTool) and obj is not BaseTool:
                        try:
                            if hasattr(obj, '__init__'):
                                sig = inspect.signature(obj.__init__)
                                params = sig.parameters
                                if len(params) <= 1 or all(
                                    p.default is not inspect.Parameter.empty 
                                    for name, p in params.items() 
                                    if name != 'self'
                                ):
                                    tool_instance = obj()
                                    self.register_tool(tool_instance)
                                    registered_count += 1
                        except Exception:
                            continue
                            
            except Exception:
                continue
        
        return registered_count
    
    def register_from_class(self, tool_class: Type[BaseTool], *args, **kwargs) -> BaseTool:
        tool_instance = tool_class(*args, **kwargs)
        self.register_tool(tool_instance)
        return tool_instance
    
    def clear(self) -> None:
        self._registry.clear()
