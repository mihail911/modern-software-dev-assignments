from typing import NoReturn, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import asc, desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Note
from ..schemas import NoteCreate, NotePatch, NoteRead

router = APIRouter(prefix="/notes", tags=["notes"])


def _raise_db_error() -> NoReturn:
    raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=list[NoteRead])
def list_notes(
    db: Session = Depends(get_db),
    q: Optional[str] = None,
    skip: int = 0,
    limit: int = Query(50, le=200),
    sort: str = Query("-created_at", description="Sort by field, prefix with - for desc"),
) -> list[NoteRead]:
    try:
        stmt = select(Note)
        if q:
            stmt = stmt.where((Note.title.contains(q)) | (Note.content.contains(q)))

        sort_field = sort.lstrip("-")
        order_fn = desc if sort.startswith("-") else asc
        if hasattr(Note, sort_field):
            stmt = stmt.order_by(order_fn(getattr(Note, sort_field)))
        else:
            stmt = stmt.order_by(desc(Note.created_at))

        rows = db.execute(stmt.offset(skip).limit(limit)).scalars().all()
    except SQLAlchemyError:
        db.rollback()
        _raise_db_error()
    return [NoteRead.model_validate(row) for row in rows]


@router.post("/", response_model=NoteRead, status_code=201)
def create_note(payload: NoteCreate, db: Session = Depends(get_db)) -> NoteRead:
    try:
        note = Note(title=payload.title, content=payload.content)
        db.add(note)
        db.flush()
        db.refresh(note)
    except SQLAlchemyError:
        db.rollback()
        _raise_db_error()
    return NoteRead.model_validate(note)


@router.patch("/{note_id}", response_model=NoteRead)
def patch_note(note_id: int, payload: NotePatch, db: Session = Depends(get_db)) -> NoteRead:
    try:
        note = db.get(Note, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        if payload.title is not None:
            note.title = payload.title
        if payload.content is not None:
            note.content = payload.content
        db.add(note)
        db.flush()
        db.refresh(note)
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        _raise_db_error()
    return NoteRead.model_validate(note)


@router.get("/{note_id}", response_model=NoteRead)
def get_note(note_id: int, db: Session = Depends(get_db)) -> NoteRead:
    try:
        note = db.get(Note, note_id)
    except SQLAlchemyError:
        db.rollback()
        _raise_db_error()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteRead.model_validate(note)


@router.delete("/{note_id}", status_code=204)
def delete_note(note_id: int, db: Session = Depends(get_db)) -> Response:
    try:
        note = db.get(Note, note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        db.delete(note)
        db.flush()
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        _raise_db_error()
    return Response(status_code=204)
