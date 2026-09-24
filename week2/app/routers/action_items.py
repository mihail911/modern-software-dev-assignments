"""Action-item endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from .. import db
from ..errors import NotFoundError
from ..schemas import (
    ActionItem,
    ActionItemDoneResponse,
    ActionItemDoneUpdate,
    ExtractedActionItem,
    ExtractRequest,
    ExtractResponse,
)
from ..services.extract import extract_action_items

router = APIRouter(prefix="/action-items", tags=["action-items"])


@router.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest) -> ExtractResponse:
    """Extract action items from notes and persist them."""
    note_id: int | None = None
    if payload.save_note:
        note_id = db.insert_note(payload.text).id

    records = db.insert_action_items(extract_action_items(payload.text), note_id=note_id)
    return ExtractResponse(
        note_id=note_id,
        items=[ExtractedActionItem(id=record.id, text=record.text) for record in records],
    )


@router.get("", response_model=list[ActionItem])
def list_all(
    note_id: int | None = Query(default=None, description="Filter by note id."),
) -> list[ActionItem]:
    """List action items, optionally filtered by their parent note."""
    return [ActionItem.model_validate(record) for record in db.list_action_items(note_id=note_id)]


@router.post(
    "/{action_item_id}/done",
    response_model=ActionItemDoneResponse,
    status_code=status.HTTP_200_OK,
)
def mark_done(action_item_id: int, payload: ActionItemDoneUpdate) -> ActionItemDoneResponse:
    """Mark an action item as done (or not done)."""
    updated = db.mark_action_item_done(action_item_id, payload.done)
    if not updated:
        raise NotFoundError(f"Action item {action_item_id} not found")
    return ActionItemDoneResponse(id=action_item_id, done=payload.done)
