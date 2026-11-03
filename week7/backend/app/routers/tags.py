from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from ..db import get_db
from ..models import Note, Tag
from ..schemas import NoteRead, TagCreate, TagPatch, TagRead

router = APIRouter(prefix="/tags", tags=["tags"])


def get_or_create_tags(db: Session, names: list[str]) -> list[Tag]:
    tags: list[Tag] = []
    seen: set[str] = set()
    for raw in names:
        name = raw.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        tag = db.execute(select(Tag).where(Tag.name == name)).scalar_one_or_none()
        if tag is None:
            tag = Tag(name=name)
            db.add(tag)
            db.flush()
        tags.append(tag)
    return tags


@router.get("/", response_model=list[TagRead])
def list_tags(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = Query(50, le=200),
    sort: str = Query("name"),
) -> list[TagRead]:
    stmt = select(Tag)
    sort_field = sort.lstrip("-")
    order_fn = desc if sort.startswith("-") else asc
    if hasattr(Tag, sort_field) and sort_field != "notes":
        stmt = stmt.order_by(order_fn(getattr(Tag, sort_field)))
    else:
        stmt = stmt.order_by(asc(Tag.name))
    rows = db.execute(stmt.offset(skip).limit(limit)).scalars().all()
    return [TagRead.model_validate(row) for row in rows]


@router.post("/", response_model=TagRead, status_code=201)
def create_tag(payload: TagCreate, db: Session = Depends(get_db)) -> TagRead:
    name = payload.name.strip().lower()
    existing = db.execute(select(Tag).where(Tag.name == name)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Tag already exists")
    tag = Tag(name=name)
    db.add(tag)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Tag already exists") from None
    db.refresh(tag)
    return TagRead.model_validate(tag)


@router.get("/{tag_id}", response_model=TagRead)
def get_tag(tag_id: int, db: Session = Depends(get_db)) -> TagRead:
    tag = db.get(Tag, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return TagRead.model_validate(tag)


@router.get("/{tag_id}/notes", response_model=list[NoteRead])
def list_notes_for_tag(tag_id: int, db: Session = Depends(get_db)) -> list[NoteRead]:
    tag = db.execute(
        select(Tag).options(selectinload(Tag.notes).selectinload(Note.tags)).where(Tag.id == tag_id)
    ).scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return [NoteRead.model_validate(note) for note in tag.notes]


@router.patch("/{tag_id}", response_model=TagRead)
def patch_tag(tag_id: int, payload: TagPatch, db: Session = Depends(get_db)) -> TagRead:
    tag = db.get(Tag, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    if payload.name is not None:
        name = payload.name.strip().lower()
        duplicate = db.execute(select(Tag).where(Tag.name == name, Tag.id != tag_id)).scalar_one_or_none()
        if duplicate:
            raise HTTPException(status_code=409, detail="Tag already exists")
        tag.name = name
    db.add(tag)
    db.flush()
    db.refresh(tag)
    return TagRead.model_validate(tag)


@router.delete("/{tag_id}", status_code=204)
def delete_tag(tag_id: int, db: Session = Depends(get_db)) -> None:
    tag = db.get(Tag, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    tag.notes.clear()
    db.delete(tag)
    db.flush()
    return None
