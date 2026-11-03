from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session, selectinload

from ..db import get_db
from ..models import Note, Tag
from ..schemas import NoteCreate, NotePatch, NoteRead, TagCreate
from .tags import get_or_create_tags

router = APIRouter(prefix="/notes", tags=["notes"])


@router.get("/", response_model=list[NoteRead])
def list_notes(
    db: Session = Depends(get_db),
    q: Optional[str] = None,
    tag: Optional[str] = None,
    skip: int = 0,
    limit: int = Query(50, le=200),
    sort: str = Query("-created_at", description="Sort by field, prefix with - for desc"),
) -> list[NoteRead]:
    stmt = select(Note).options(selectinload(Note.tags))
    if q:
        stmt = stmt.where((Note.title.contains(q)) | (Note.content.contains(q)))
    if tag:
        stmt = stmt.join(Note.tags).where(Tag.name == tag.strip().lower()).distinct()

    sort_field = sort.lstrip("-")
    order_fn = desc if sort.startswith("-") else asc
    if hasattr(Note, sort_field) and sort_field != "tags":
        stmt = stmt.order_by(order_fn(getattr(Note, sort_field)))
    else:
        stmt = stmt.order_by(desc(Note.created_at))

    rows = db.execute(stmt.offset(skip).limit(limit)).scalars().unique().all()
    return [NoteRead.model_validate(row) for row in rows]


@router.post("/", response_model=NoteRead, status_code=201)
def create_note(payload: NoteCreate, db: Session = Depends(get_db)) -> NoteRead:
    note = Note(title=payload.title, content=payload.content)
    note.tags = get_or_create_tags(db, payload.tags)
    db.add(note)
    db.flush()
    db.refresh(note)
    return NoteRead.model_validate(note)


@router.patch("/{note_id}", response_model=NoteRead)
def patch_note(note_id: int, payload: NotePatch, db: Session = Depends(get_db)) -> NoteRead:
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    if payload.title is not None:
        note.title = payload.title
    if payload.content is not None:
        note.content = payload.content
    if payload.tags is not None:
        note.tags = get_or_create_tags(db, payload.tags)
    db.add(note)
    db.flush()
    db.refresh(note)
    return NoteRead.model_validate(note)


@router.get("/{note_id}", response_model=NoteRead)
def get_note(note_id: int, db: Session = Depends(get_db)) -> NoteRead:
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteRead.model_validate(note)


@router.post("/{note_id}/tags", response_model=NoteRead)
def add_tag_to_note(note_id: int, payload: TagCreate, db: Session = Depends(get_db)) -> NoteRead:
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    tags = get_or_create_tags(db, [payload.name])
    if not tags:
        raise HTTPException(status_code=422, detail="Tag name is required")
    tag = tags[0]
    if tag not in note.tags:
        note.tags.append(tag)
    db.add(note)
    db.flush()
    db.refresh(note)
    return NoteRead.model_validate(note)


@router.delete("/{note_id}/tags/{tag_id}", response_model=NoteRead)
def remove_tag_from_note(note_id: int, tag_id: int, db: Session = Depends(get_db)) -> NoteRead:
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    tag = db.get(Tag, tag_id)
    if not tag or tag not in note.tags:
        raise HTTPException(status_code=404, detail="Tag not found on note")
    note.tags.remove(tag)
    db.add(note)
    db.flush()
    db.refresh(note)
    return NoteRead.model_validate(note)


