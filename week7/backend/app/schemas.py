from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def normalize(self) -> "NoteCreate":
        self.title = self.title.strip()
        self.content = self.content.strip()
        if not self.title:
            raise ValueError("title must not be empty")
        if not self.content:
            raise ValueError("content must not be empty")
        return self


class NoteRead(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NotePatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_patch(self) -> "NotePatch":
        if self.title is None and self.content is None:
            raise ValueError("at least one field must be provided")
        if self.title is not None:
            self.title = self.title.strip()
            if not self.title:
                raise ValueError("title must not be empty")
        if self.content is not None:
            self.content = self.content.strip()
            if not self.content:
                raise ValueError("content must not be empty")
        return self


class ActionItemCreate(BaseModel):
    description: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def normalize(self) -> "ActionItemCreate":
        self.description = self.description.strip()
        if not self.description:
            raise ValueError("description must not be empty")
        return self


class ActionItemRead(BaseModel):
    id: int
    description: str
    completed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ActionItemPatch(BaseModel):
    description: str | None = Field(default=None, min_length=1, max_length=10_000)
    completed: bool | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> "ActionItemPatch":
        if self.description is None and self.completed is None:
            raise ValueError("at least one field must be provided")
        if self.description is not None:
            self.description = self.description.strip()
            if not self.description:
                raise ValueError("description must not be empty")
        return self


