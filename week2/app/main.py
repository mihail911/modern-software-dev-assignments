"""FastAPI application entrypoint.

The app is built by :func:`create_app` so tests can construct isolated
instances, and the module-level ``app`` keeps `uvicorn week2.app.main:app`
working. Database initialisation runs in the lifespan handler rather than at
import time, so importing the module has no side effects.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import init_db
from .errors import register_exception_handlers
from .routers import action_items, notes

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI) -> Iterator[None]:
    """Initialise storage on startup and release resources on shutdown."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
    )

    register_exception_handlers(application)

    application.include_router(notes.router)
    application.include_router(action_items.router)

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    application.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    return application


app = create_app()
