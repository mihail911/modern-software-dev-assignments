import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capabilities.tool_manager import ToolManager
from src.capabilities.resource_manager import ResourceManager, Resource
from src.capabilities.prompt_manager import PromptManager, Prompt, PromptArgument
from src.capabilities.capability_layer import CapabilityLayer
from src.tools.base import BaseTool, ToolRegistry


class MockTool(BaseTool):
    def __init__(self, name: str = "mock_tool"):
        super().__init__(name=name, description=f"A mock tool: {name}")
    
    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "Test value"}
            },
            "required": ["value"]
        }
    
    async def execute(self, value: str) -> dict[str, Any]:
        return {"success": True, "result": value.upper()}


class TestToolManager:
    def setup_method(self):
        ToolRegistry.reset()
        self.manager = ToolManager()
    
    def test_initialization(self):
        assert self.manager.registry is not None
    
    def test_register_tool(self):
        tool = MockTool()
        self.manager.register_tool(tool)
        
        assert self.manager.get_tool("mock_tool") is tool
    
    def test_unregister_tool(self):
        tool = MockTool()
        self.manager.register_tool(tool)
        
        self.manager.unregister_tool("mock_tool")
        assert self.manager.get_tool("mock_tool") is None
    
    def test_list_tools(self):
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")
        
        self.manager.register_tool(tool1)
        self.manager.register_tool(tool2)
        
        tools = self.manager.list_tools()
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
    
    def test_list_tool_names(self):
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")
        
        self.manager.register_tool(tool1)
        self.manager.register_tool(tool2)
        
        names = self.manager.list_tool_names()
        assert len(names) == 2
        assert "tool1" in names
        assert "tool2" in names
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        tool = MockTool()
        self.manager.register_tool(tool)
        
        result = await self.manager.execute_tool("mock_tool", value="hello")
        
        assert result["success"] is True
        assert result["result"] == "HELLO"
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        result = await self.manager.execute_tool("nonexistent", value="test")
        
        assert result["success"] is False
        assert "not found" in result["error"]
    
    def test_register_from_class(self):
        tool = self.manager.register_from_class(MockTool)
        
        assert isinstance(tool, MockTool)
        assert self.manager.get_tool("mock_tool") is tool
    
    def test_clear(self):
        tool = MockTool()
        self.manager.register_tool(tool)
        
        self.manager.clear()
        assert len(self.manager.list_tools()) == 0


class TestResourceManager:
    def setup_method(self):
        self.manager = ResourceManager()
    
    def test_register_static_resource(self):
        resource = self.manager.register_static_resource(
            uri="test://static",
            name="Test Static",
            description="A static test resource",
            content="Hello World",
            mime_type="text/plain"
        )
        
        assert resource.uri == "test://static"
        assert resource.name == "Test Static"
        assert resource.content == "Hello World"
        assert resource.is_static is True
    
    @pytest.mark.asyncio
    async def test_register_dynamic_resource(self):
        async def generator():
            return {"dynamic": "data"}
        
        resource = self.manager.register_dynamic_resource(
            uri="test://dynamic",
            name="Test Dynamic",
            description="A dynamic test resource",
            generator=generator,
            mime_type="application/json"
        )
        
        assert resource.uri == "test://dynamic"
        assert resource.is_static is False
        
        content = await resource.get_content()
        assert content == {"dynamic": "data"}
    
    def test_register_file_resource(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "value"}, f)
            temp_path = Path(f.name)
        
        try:
            resource = self.manager.register_file_resource(
                uri="file://test",
                file_path=temp_path,
                name="Test File",
                description="Test file resource"
            )
            
            assert resource is not None
            assert resource.uri == "file://test"
            assert resource.mime_type == "application/json"
            assert resource.content == {"test": "value"}
        finally:
            temp_path.unlink()
    
    def test_register_file_resource_not_found(self):
        resource = self.manager.register_file_resource(
            uri="file://nonexistent",
            file_path=Path("/nonexistent/file.txt")
        )
        
        assert resource is None
    
    def test_unregister_resource(self):
        self.manager.register_static_resource(
            uri="test://remove",
            name="To Remove",
            description="",
            content="content"
        )
        
        self.manager.unregister_resource("test://remove")
        assert self.manager.get_resource("test://remove") is None
    
    def test_get_resource(self):
        self.manager.register_static_resource(
            uri="test://get",
            name="Get Test",
            description="",
            content="test content"
        )
        
        resource = self.manager.get_resource("test://get")
        assert resource is not None
        assert resource.name == "Get Test"
    
    @pytest.mark.asyncio
    async def test_get_resource_content_static(self):
        self.manager.register_static_resource(
            uri="test://content",
            name="Content Test",
            description="",
            content="static content"
        )
        
        content = await self.manager.get_resource_content("test://content")
        assert content == "static content"
    
    @pytest.mark.asyncio
    async def test_get_resource_content_dynamic(self):
        call_count = [0]
        
        async def generator():
            call_count[0] += 1
            return {"call": call_count[0]}
        
        self.manager.register_dynamic_resource(
            uri="test://dynamic-content",
            name="Dynamic Content",
            description="",
            generator=generator
        )
        
        content1 = await self.manager.get_resource_content("test://dynamic-content")
        content2 = await self.manager.get_resource_content("test://dynamic-content")
        
        assert content1 == {"call": 1}
        assert content2 == {"call": 2}
    
    @pytest.mark.asyncio
    async def test_get_resource_content_not_found(self):
        content = await self.manager.get_resource_content("test://nonexistent")
        assert content is None
    
    def test_list_resources(self):
        self.manager.register_static_resource(
            uri="test://list1",
            name="List 1",
            description="First",
            content="1"
        )
        self.manager.register_static_resource(
            uri="test://list2",
            name="List 2",
            description="Second",
            content="2"
        )
        
        resources = self.manager.list_resources()
        assert len(resources) == 2
        uris = [r["uri"] for r in resources]
        assert "test://list1" in uris
        assert "test://list2" in uris
    
    def test_update_resource_content(self):
        self.manager.register_static_resource(
            uri="test://update",
            name="Update Test",
            description="",
            content="original"
        )
        
        result = self.manager.update_resource_content("test://update", "updated")
        assert result is True
        
        resource = self.manager.get_resource("test://update")
        assert resource.content == "updated"
    
    def test_update_resource_content_dynamic(self):
        async def generator():
            return {}
        
        self.manager.register_dynamic_resource(
            uri="test://dynamic-update",
            name="Dynamic Update",
            description="",
            generator=generator
        )
        
        result = self.manager.update_resource_content("test://dynamic-update", "updated")
        assert result is False
    
    def test_clear(self):
        self.manager.register_static_resource(
            uri="test://clear",
            name="Clear Test",
            description="",
            content="content"
        )
        
        self.manager.clear()
        assert len(self.manager.list_resources()) == 0


class TestPromptManager:
    def setup_method(self):
        self.manager = PromptManager()
    
    def test_register_prompt(self):
        prompt = self.manager.register_prompt(
            name="test_prompt",
            description="A test prompt",
            template="Hello {name}!",
            arguments=[
                PromptArgument(name="name", description="The name to greet", required=True)
            ]
        )
        
        assert prompt.name == "test_prompt"
        assert prompt.description == "A test prompt"
        assert prompt.template == "Hello {name}!"
        assert len(prompt.arguments) == 1
    
    def test_get_prompt(self):
        self.manager.register_prompt(
            name="get_test",
            description="Test",
            template="Template"
        )
        
        prompt = self.manager.get_prompt("get_test")
        assert prompt is not None
        assert prompt.name == "get_test"
    
    def test_get_prompt_not_found(self):
        prompt = self.manager.get_prompt("nonexistent")
        assert prompt is None
    
    def test_render_prompt(self):
        self.manager.register_prompt(
            name="render_test",
            description="Test",
            template="Hello {name}! You are {age} years old.",
            arguments=[
                PromptArgument(name="name", description="Name", required=True),
                PromptArgument(name="age", description="Age", required=True)
            ]
        )
        
        result = self.manager.render_prompt("render_test", name="Alice", age=30)
        assert result == "Hello Alice! You are 30 years old."
    
    def test_render_prompt_missing_required(self):
        self.manager.register_prompt(
            name="missing_arg",
            description="Test",
            template="Hello {name}!",
            arguments=[
                PromptArgument(name="name", description="Name", required=True)
            ]
        )
        
        with pytest.raises(ValueError, match="Required argument 'name' not provided"):
            self.manager.render_prompt("missing_arg")
    
    def test_render_prompt_with_default(self):
        self.manager.register_prompt(
            name="default_test",
            description="Test",
            template="Hello {name}! Language: {lang}",
            arguments=[
                PromptArgument(name="name", description="Name", required=True),
                PromptArgument(name="lang", description="Language", required=False, default="English")
            ]
        )
        
        result = self.manager.render_prompt("default_test", name="Bob")
        assert result == "Hello Bob! Language: English"
    
    def test_list_prompts(self):
        self.manager.register_prompt(
            name="prompt1",
            description="First prompt",
            template="Template 1"
        )
        self.manager.register_prompt(
            name="prompt2",
            description="Second prompt",
            template="Template 2"
        )
        
        prompts = self.manager.list_prompts()
        assert len(prompts) == 2
        names = [p["name"] for p in prompts]
        assert "prompt1" in names
        assert "prompt2" in names
    
    def test_unregister_prompt(self):
        self.manager.register_prompt(
            name="remove_me",
            description="",
            template=""
        )
        
        self.manager.unregister_prompt("remove_me")
        assert self.manager.get_prompt("remove_me") is None
    
    def test_update_prompt(self):
        self.manager.register_prompt(
            name="update_test",
            description="Original description",
            template="Original template",
            arguments=[
                PromptArgument(name="old", description="Old arg", required=True)
            ]
        )
        
        result = self.manager.update_prompt(
            name="update_test",
            description="Updated description",
            template="Updated template: {new}",
            arguments=[
                PromptArgument(name="new", description="New arg", required=True)
            ]
        )
        
        assert result is True
        
        prompt = self.manager.get_prompt("update_test")
        assert prompt.description == "Updated description"
        assert prompt.template == "Updated template: {new}"
        assert len(prompt.arguments) == 1
        assert prompt.arguments[0].name == "new"
    
    def test_update_prompt_not_found(self):
        result = self.manager.update_prompt("nonexistent", template="test")
        assert result is False
    
    def test_clear(self):
        self.manager.register_prompt(
            name="clear_test",
            description="",
            template=""
        )
        
        self.manager.clear()
        assert len(self.manager.list_prompts()) == 0


class TestCapabilityLayer:
    def setup_method(self):
        ToolRegistry.reset()
        self.layer = CapabilityLayer()
    
    def test_initialization(self):
        assert self.layer.tool_manager is not None
        assert self.layer.resource_manager is not None
        assert self.layer.prompt_manager is not None
    
    def test_initialize(self):
        result = self.layer.initialize(auto_register=False)
        
        assert result["success"] is True
        assert "Capability layer initialized" in result["message"]
        assert self.layer._initialized is True
    
    def test_initialize_already_initialized(self):
        self.layer.initialize(auto_register=False)
        result = self.layer.initialize(auto_register=False)
        
        assert result["success"] is True
        assert "already initialized" in result["message"]
    
    def test_get_capabilities(self):
        tool = MockTool()
        self.layer.register_tool(tool)
        
        self.layer.resource_manager.register_static_resource(
            uri="test://cap",
            name="Test",
            description="",
            content=""
        )
        
        self.layer.prompt_manager.register_prompt(
            name="test_cap",
            description="",
            template=""
        )
        
        caps = self.layer.get_capabilities()
        
        assert caps["tools"]["enabled"] is True
        assert caps["tools"]["count"] == 1
        assert "mock_tool" in caps["tools"]["names"]
        
        assert caps["resources"]["enabled"] is True
        assert caps["resources"]["count"] == 1
        
        assert caps["prompts"]["enabled"] is True
        assert caps["prompts"]["count"] == 1
    
    def test_list_tools(self):
        tool = MockTool(name="cap_tool")
        self.layer.register_tool(tool)
        
        tools = self.layer.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "cap_tool"
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        tool = MockTool()
        self.layer.register_tool(tool)
        
        result = await self.layer.execute_tool("mock_tool", value="test")
        
        assert result["success"] is True
        assert result["result"] == "TEST"
    
    def test_list_resources(self):
        self.layer.resource_manager.register_static_resource(
            uri="cap://test",
            name="Cap Resource",
            description="",
            content=""
        )
        
        resources = self.layer.list_resources()
        assert len(resources) == 1
        assert resources[0]["uri"] == "cap://test"
    
    @pytest.mark.asyncio
    async def test_get_resource(self):
        self.layer.resource_manager.register_static_resource(
            uri="cap://get",
            name="",
            description="",
            content="resource content"
        )
        
        content = await self.layer.get_resource("cap://get")
        assert content == "resource content"
    
    def test_list_prompts(self):
        self.layer.prompt_manager.register_prompt(
            name="cap_prompt",
            description="",
            template=""
        )
        
        prompts = self.layer.list_prompts()
        assert len(prompts) == 1
        assert prompts[0]["name"] == "cap_prompt"
    
    def test_render_prompt(self):
        self.layer.prompt_manager.register_prompt(
            name="cap_render",
            description="",
            template="Hello {name}!",
            arguments=[
                PromptArgument(name="name", description="", required=True)
            ]
        )
        
        result = self.layer.render_prompt("cap_render", name="World")
        assert result == "Hello World!"
    
    def test_shutdown(self):
        tool = MockTool()
        self.layer.register_tool(tool)
        
        self.layer.resource_manager.register_static_resource(
            uri="shutdown://test",
            name="",
            description="",
            content=""
        )
        
        self.layer.prompt_manager.register_prompt(
            name="shutdown_prompt",
            description="",
            template=""
        )
        
        self.layer.shutdown()
        
        assert len(self.layer.list_tools()) == 0
        assert len(self.layer.list_resources()) == 0
        assert len(self.layer.list_prompts()) == 0
        assert self.layer._initialized is False
