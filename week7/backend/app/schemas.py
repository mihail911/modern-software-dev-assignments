from datetime import datetime

from pydantic import BaseModel, Field, field_validator


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank or whitespace-only")
    return value


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)

    @field_validator("title", "content")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        return _reject_blank(value)


class NoteRead(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NotePatch(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    content: str | None = Field(default=None, min_length=1)

    @field_validator("title", "content")
    @classmethod
    def reject_blank_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _reject_blank(value)


class ActionItemCreate(BaseModel):
    description: str = Field(..., min_length=1)

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        return _reject_blank(value)


class ActionItemRead(BaseModel):
    id: int
    description: str
    completed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ActionItemPatch(BaseModel):
    description: str | None = Field(default=None, min_length=1)
    completed: bool | None = None

    @field_validator("description")
    @classmethod
    def reject_blank_optional_description(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _reject_blank(value)
