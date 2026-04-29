from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    input_schema: dict[str, Any] = Field(..., description="输入参数的JSON Schema定义")


class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @abstractmethod
    def get_input_schema(self) -> dict[str, Any]:
        pass
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.get_input_schema()
        )
    
    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        pass


class ToolRegistry:
    _instance: Optional["ToolRegistry"] = None
    _tools: dict[str, BaseTool] = {}
    
    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._tools = {}
    
    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> None:
        if tool_name in self._tools:
            del self._tools[tool_name]
    
    def get(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)
    
    def get_all(self) -> dict[str, BaseTool]:
        return self._tools.copy()
    
    def list_definitions(self) -> list[ToolDefinition]:
        return [tool.get_definition() for tool in self._tools.values()]
    
    def clear(self) -> None:
        self._tools.clear()
