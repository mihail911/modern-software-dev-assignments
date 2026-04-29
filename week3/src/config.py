import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


class Config:
    _instance: Optional["Config"] = None
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
        
        self._initialized = True
    
    @property
    def openweathermap_api_key(self) -> Optional[str]:
        return os.getenv("OPENWEATHERMAP_API_KEY")
    
    @property
    def newsapi_key(self) -> Optional[str]:
        return os.getenv("NEWSAPI_KEY")
    
    @property
    def mcp_server_name(self) -> str:
        return os.getenv("MCP_SERVER_NAME", "custom-mcp-server")
    
    @property
    def mcp_server_version(self) -> str:
        return os.getenv("MCP_SERVER_VERSION", "1.0.0")
    
    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO").upper()
    
    @classmethod
    def reset(cls) -> None:
        cls._instance = None


config = Config()
