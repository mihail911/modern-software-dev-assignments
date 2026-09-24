"""SQLite data-access layer.

This module owns every SQL statement in the application. Callers receive plain
:class:`NoteRecord` / :class:`ActionItemRecord` dataclasses rather than raw
``sqlite3.Row`` objects, so the rest of the codebase is decoupled from the
storage engine and free of index-based/``row["col"]`` access.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from .config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER REFERENCES notes(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_action_items_note_id ON action_items(note_id);
"""


@dataclass(frozen=True)
class NoteRecord:
    """A persisted note."""

    id: int
    content: str
    created_at: str | None


@dataclass(frozen=True)
class ActionItemRecord:
    """A persisted action item."""

    id: int
    note_id: int | None
    text: str
    done: bool
    created_at: str | None


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection wrapped in a committing/rolling-back transaction.

    Foreign keys are enabled per connection (SQLite defaults them off), and the
    connection is always closed when the context exits.
    """
    settings = get_settings()
    settings.ensure_data_directory()
    connection = sqlite3.connect(settings.db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the schema if it does not already exist."""
    with get_connection() as connection:
        connection.executescript(_SCHEMA)


def _note_from_row(row: sqlite3.Row) -> NoteRecord:
    return NoteRecord(id=row["id"], content=row["content"], created_at=row["created_at"])


def _action_item_from_row(row: sqlite3.Row) -> ActionItemRecord:
    return ActionItemRecord(
        id=row["id"],
        note_id=row["note_id"],
        text=row["text"],
        done=bool(row["done"]),
        created_at=row["created_at"],
    )


def insert_note(content: str) -> NoteRecord:
    """Insert a note and return the persisted record."""
    with get_connection() as connection:
        cursor = connection.execute("INSERT INTO notes (content) VALUES (?)", (content,))
        note_id = int(cursor.lastrowid)
        row = connection.execute(
            "SELECT id, content, created_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    return _note_from_row(row)


def get_note(note_id: int) -> NoteRecord | None:
    """Return a single note, or ``None`` if it does not exist."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, content, created_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    return _note_from_row(row) if row is not None else None


def list_notes() -> list[NoteRecord]:
    """Return all notes, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, content, created_at FROM notes ORDER BY id DESC"
        ).fetchall()
    return [_note_from_row(row) for row in rows]


def insert_action_items(items: Sequence[str], note_id: int | None = None) -> list[ActionItemRecord]:
    """Insert action items in a single transaction and return them.

    New items share the same default ``done=False`` state; the records are read
    back so callers get server-assigned ids and timestamps.
    """
    items = [item for item in items if item and item.strip()]
    if not items:
        return []

    with get_connection() as connection:
        ids: list[int] = []
        for item in items:
            cursor = connection.execute(
                "INSERT INTO action_items (note_id, text) VALUES (?, ?)",
                (note_id, item),
            )
            ids.append(int(cursor.lastrowid))

        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT id, note_id, text, done, created_at FROM action_items "
            f"WHERE id IN ({placeholders}) ORDER BY id",
            ids,
        ).fetchall()

    rows_by_id = {row["id"]: row for row in rows}
    return [_action_item_from_row(rows_by_id[item_id]) for item_id in ids]


def list_action_items(note_id: int | None = None) -> list[ActionItemRecord]:
    """Return action items, optionally filtered by note, newest first."""
    query = "SELECT id, note_id, text, done, created_at FROM action_items"
    params: tuple[object, ...] = ()
    if note_id is not None:
        query += " WHERE note_id = ?"
        params = (note_id,)
    query += " ORDER BY id DESC"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [_action_item_from_row(row) for row in rows]


def get_action_item(action_item_id: int) -> ActionItemRecord | None:
    """Return a single action item, or ``None`` if it does not exist."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, note_id, text, done, created_at FROM action_items WHERE id = ?",
            (action_item_id,),
        ).fetchone()
    return _action_item_from_row(row) if row is not None else None


def mark_action_item_done(action_item_id: int, done: bool) -> bool:
    """Set the ``done`` flag. Return ``False`` if the item does not exist."""
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE action_items SET done = ? WHERE id = ?",
            (1 if done else 0, action_item_id),
        )
        return cursor.rowcount > 0
