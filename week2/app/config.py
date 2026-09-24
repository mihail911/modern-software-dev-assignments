"""Application configuration.

Configuration is centralised here instead of being scattered across modules and
read at import time. Values come from environment variables with sensible
defaults so the app works out of the box, while still being easy to point at a
different database or Ollama model in tests/deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Project layout: ``app/config.py`` -> parents[1] is the ``week2`` directory.
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "app.db"


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings."""

    app_title: str
    app_version: str
    data_dir: Path
    db_path: Path
    ollama_model: str
    ollama_base_url: str
    llm_temperature: float

    def ensure_data_directory(self) -> None:
        """Create the directory that holds the SQLite database if needed."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    data_dir = Path(os.getenv("APP_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    db_path = Path(os.getenv("APP_DB_PATH", str(data_dir / "app.db"))).expanduser()
    return Settings(
        app_title=os.getenv("APP_TITLE", "Action Item Extractor"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        data_dir=data_dir,
        db_path=db_path,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        llm_temperature=_env_float("OLLAMA_TEMPERATURE", 0.0),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return load_settings()
