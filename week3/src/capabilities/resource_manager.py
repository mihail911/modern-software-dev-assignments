import json
from pathlib import Path
from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    content: Any = None
    is_static: bool = True
    last_modified: Optional[datetime] = None
    _dynamic_generator: Optional[Callable[[], Awaitable[Any]]] = None
    
    async def get_content(self) -> Any:
        if self._dynamic_generator is not None:
            return await self._dynamic_generator()
        return self.content


class ResourceManager:
    def __init__(self):
        self._resources: dict[str, Resource] = {}
        self._subscribers: dict[str, list[Callable[[Resource], Awaitable[None]]]] = {}
    
    def register_static_resource(
        self,
        uri: str,
        name: str,
        description: str,
        content: Any,
        mime_type: str = "text/plain"
    ) -> Resource:
        resource = Resource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            content=content,
            is_static=True,
            last_modified=datetime.now()
        )
        self._resources[uri] = resource
        return resource
    
    def register_dynamic_resource(
        self,
        uri: str,
        name: str,
        description: str,
        generator: Callable[[], Awaitable[Any]],
        mime_type: str = "application/json"
    ) -> Resource:
        resource = Resource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            is_static=False,
            last_modified=datetime.now(),
            _dynamic_generator=generator
        )
        self._resources[uri] = resource
        return resource
    
    def register_file_resource(
        self,
        uri: str,
        file_path: Path,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Resource]:
        if not file_path.exists():
            return None
        
        if not file_path.is_file():
            return None
        
        mime_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".py": "text/x-python",
            ".yaml": "text/yaml",
            ".yml": "text/yaml"
        }
        
        suffix = file_path.suffix.lower()
        mime_type = mime_types.get(suffix, "application/octet-stream")
        
        try:
            if mime_type == "application/json":
                content = json.loads(file_path.read_text(encoding="utf-8"))
            else:
                content = file_path.read_text(encoding="utf-8")
        except Exception:
            return None
        
        resource = Resource(
            uri=uri,
            name=name or file_path.name,
            description=description or f"File resource: {file_path.name}",
            mime_type=mime_type,
            content=content,
            is_static=True,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
        self._resources[uri] = resource
        return resource
    
    def unregister_resource(self, uri: str) -> None:
        if uri in self._resources:
            del self._resources[uri]
        if uri in self._subscribers:
            del self._subscribers[uri]
    
    def get_resource(self, uri: str) -> Optional[Resource]:
        return self._resources.get(uri)
    
    async def get_resource_content(self, uri: str) -> Optional[Any]:
        resource = self._resources.get(uri)
        if resource is None:
            return None
        return await resource.get_content()
    
    def list_resources(self) -> list[dict[str, Any]]:
        return [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
                "mime_type": r.mime_type,
                "is_static": r.is_static,
                "last_modified": r.last_modified.isoformat() if r.last_modified else None
            }
            for r in self._resources.values()
        ]
    
    def subscribe(self, uri: str, callback: Callable[[Resource], Awaitable[None]]) -> None:
        if uri not in self._subscribers:
            self._subscribers[uri] = []
        self._subscribers[uri].append(callback)
    
    def unsubscribe(self, uri: str, callback: Callable[[Resource], Awaitable[None]]) -> None:
        if uri in self._subscribers:
            self._subscribers[uri].remove(callback)
    
    async def notify_subscribers(self, uri: str) -> None:
        if uri not in self._subscribers or uri not in self._resources:
            return
        
        resource = self._resources[uri]
        for callback in self._subscribers[uri]:
            await callback(resource)
    
    def update_resource_content(self, uri: str, content: Any) -> bool:
        if uri not in self._resources:
            return False
        
        resource = self._resources[uri]
        if not resource.is_static:
            return False
        
        resource.content = content
        resource.last_modified = datetime.now()
        return True
    
    def clear(self) -> None:
        self._resources.clear()
        self._subscribers.clear()
