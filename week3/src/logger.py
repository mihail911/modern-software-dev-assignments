import sys
import logging
from typing import Optional
from .config import config


class StderrLogger:
    _instance: Optional["StderrLogger"] = None
    
    def __new__(cls) -> "StderrLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._logger = logging.getLogger("mcp-server")
        self._logger.setLevel(getattr(logging, config.log_level, logging.INFO))
        
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(getattr(logging, config.log_level, logging.INFO))
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        self._logger.addHandler(handler)
        self._initialized = True
    
    def debug(self, message: str):
        self._logger.debug(message)
    
    def info(self, message: str):
        self._logger.info(message)
    
    def warning(self, message: str):
        self._logger.warning(message)
    
    def error(self, message: str):
        self._logger.error(message)
    
    def critical(self, message: str):
        self._logger.critical(message)
    
    @classmethod
    def reset(cls) -> None:
        cls._instance = None


logger = StderrLogger()
