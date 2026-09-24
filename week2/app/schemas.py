"""Pydantic models that define the API request/response contracts.

Keeping these in one place gives the FastAPI routes and the generated OpenAPI
documentation a single source of truth for validation and serialisation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NoteCreate(BaseModel):
    """Request body for creating a note."""

    content: str = Field(..., min_length=1, description="Free-form note content.")

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content must not be blank")
        return cleaned


class Note(BaseModel):
    """A stored note returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str
    created_at: datetime | str | None = None


class ActionItem(BaseModel):
    """A stored action item returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    note_id: int | None = None
    text: str
    done: bool = False
    created_at: datetime | str | None = None


class ActionItemDoneUpdate(BaseModel):
    """Request body for toggling an action item's completion state."""

    done: bool = True


class ActionItemDoneResponse(BaseModel):
    """Response returned after updating an action item."""

    id: int
    done: bool


class ExtractRequest(BaseModel):
    """Request body for the heuristic action-item extraction endpoint."""

    text: str = Field(..., min_length=1, description="Notes to extract action items from.")
    save_note: bool = Field(
        default=False,
        description="Persist the source text as a note before extracting.",
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class ExtractedActionItem(BaseModel):
    """A single extracted action item (not necessarily persisted)."""

    id: int | None = None
    text: str


class ExtractResponse(BaseModel):
    """Response for the extraction endpoints."""

    note_id: int | None = None
    items: list[ExtractedActionItem] = Field(default_factory=list)


class ExtractedItems(BaseModel):
    """Structured-output schema requested from the LLM.

    ``model_json_schema()`` on this model is passed to Ollama's structured
    outputs API, and responses are validated back into this same type.
    """

    items: list[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    """Machine-readable error payload."""

    type: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """Consistent envelope for all error responses."""

    error: ErrorDetail
