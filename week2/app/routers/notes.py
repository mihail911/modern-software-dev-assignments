"""Note endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from .. import db
from ..errors import NotFoundError
from ..schemas import Note, NoteCreate

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post("", response_model=Note, status_code=status.HTTP_201_CREATED)
def create_note(payload: NoteCreate) -> Note:
    """Create a note from the request body."""
    record = db.insert_note(payload.content)
    return Note.model_validate(record)


@router.get("/{note_id}", response_model=Note)
def get_single_note(note_id: int) -> Note:
    """Fetch a single note by id."""
    record = db.get_note(note_id)
    if record is None:
        raise NotFoundError(f"Note {note_id} not found")
    return Note.model_validate(record)
