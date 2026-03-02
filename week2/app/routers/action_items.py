from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db
from ..services.extract import extract_action_items_llm


router = APIRouter(prefix="/action-items", tags=["action-items"])


class ExtractPayload(BaseModel):
    text: str
    save_note: Optional[bool] = False


class ActionItemOut(BaseModel):
    id: int
    text: str


class ExtractResponse(BaseModel):
    note_id: Optional[int]
    items: List[ActionItemOut]


class ActionItemRow(BaseModel):
    id: int
    note_id: Optional[int]
    text: str
    done: bool
    created_at: str


class MarkDonePayload(BaseModel):
    done: Optional[bool] = True


@router.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractPayload) -> ExtractResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    note_id: Optional[int] = None
    if payload.save_note:
        try:
            note_id = db.insert_note(text)
        except Exception:
            raise HTTPException(status_code=500, detail="failed to save note")

    # Use the LLM-backed extractor which falls back to heuristic extraction
    items = extract_action_items_llm(text)

    try:
        ids = db.insert_action_items(items, note_id=note_id)
    except Exception:
        raise HTTPException(status_code=500, detail="failed to insert action items")

    return ExtractResponse(note_id=note_id, items=[ActionItemOut(id=i, text=t) for i, t in zip(ids, items)])


@router.get("", response_model=List[ActionItemRow])
def list_all(note_id: Optional[int] = None) -> List[ActionItemRow]:
    rows = db.list_action_items(note_id=note_id)
    return [
        ActionItemRow(
            id=r["id"],
            note_id=r["note_id"],
            text=r["text"],
            done=bool(r["done"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.post("/{action_item_id}/done")
def mark_done(action_item_id: int, payload: MarkDonePayload) -> ActionItemOut:
    item = db.get_action_item(action_item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="action item not found")

    done = bool(payload.done)
    try:
        db.mark_action_item_done(action_item_id, done)
    except Exception:
        raise HTTPException(status_code=500, detail="failed to update action item")

    return ActionItemOut(id=action_item_id, text=item["text"])


