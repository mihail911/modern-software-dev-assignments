from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Notebook
from ..schemas import NotebookCreate, NotebookRead

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


@router.get("/", response_model=list[NotebookRead])
def list_notebooks(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = Query(50, le=200),
    sort: str = Query("-created_at"),
) -> list[NotebookRead]:
    stmt = select(Notebook)
    sort_field = sort.lstrip("-")
    order_fn = desc if sort.startswith("-") else asc
    if hasattr(Notebook, sort_field):
        stmt = stmt.order_by(order_fn(getattr(Notebook, sort_field)))
    else:
        stmt = stmt.order_by(desc(Notebook.created_at))

    rows = db.execute(stmt.offset(skip).limit(limit)).scalars().all()
    return [NotebookRead.model_validate(row) for row in rows]


@router.post("/", response_model=NotebookRead, status_code=201)
def create_notebook(payload: NotebookCreate, db: Session = Depends(get_db)) -> NotebookRead:
    notebook = Notebook(name=payload.name)
    db.add(notebook)
    db.flush()
    db.refresh(notebook)
    return NotebookRead.model_validate(notebook)


@router.get("/{notebook_id}", response_model=NotebookRead)
def get_notebook(notebook_id: int, db: Session = Depends(get_db)) -> NotebookRead:
    notebook = db.get(Notebook, notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return NotebookRead.model_validate(notebook)

