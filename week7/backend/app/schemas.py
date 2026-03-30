from datetime import datetime

from pydantic import BaseModel


class NotebookCreate(BaseModel):
    name: str


class NotebookRead(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NoteCreate(BaseModel):
    title: str
    content: str
    notebook_id: int | None = None


class NoteRead(BaseModel):
    id: int
    title: str
    content: str
    notebook_id: int | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NotePatch(BaseModel):
    title: str | None = None
    content: str | None = None


class ActionItemCreate(BaseModel):
    description: str
    note_id: int | None = None


class ActionItemRead(BaseModel):
    id: int
    description: str
    completed: bool
    note_id: int | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ActionItemPatch(BaseModel):
    description: str | None = None
    completed: bool | None = None


