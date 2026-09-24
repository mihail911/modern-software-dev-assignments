import json
import os
import urllib.request

import pytest

from ..app.services.extract import OLLAMA_MODEL, extract_action_items, extract_action_items_llm


# ---------------------------------------------------------------------------
# Real-LLM integration tests
# ---------------------------------------------------------------------------
# These tests intentionally do NOT mock ``ollama.chat``. They call the real
# model served by Ollama, which requires ``ollama serve`` to be running and the
# configured model (``OLLAMA_MODEL``, default ``llama3.1:8b``) to be pulled.
#
# Because real model output is non-deterministic, assertions check for the
# presence of key concepts (case-insensitive) and the function's contract
# rather than exact strings. Tests are skipped automatically when the server or
# model is unavailable, so the rest of the suite still runs offline.
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _ollama_model_ready() -> tuple[bool, str]:
    """Return (ready, reason) for whether Ollama serves the configured model."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - report any connection failure as a skip
        return False, f"Ollama server not reachable at {OLLAMA_BASE_URL}: {exc}"

    names = {model.get("name", "") for model in data.get("models", [])}
    base = OLLAMA_MODEL.split(":")[0]
    if OLLAMA_MODEL in names or base in {name.split(":")[0] for name in names}:
        return True, ""
    return False, f"Ollama model '{OLLAMA_MODEL}' is not pulled"


@pytest.fixture(scope="module")
def ollama_ready():
    """Skip real-LLM tests when Ollama/model is unavailable."""
    ready, reason = _ollama_model_ready()
    if not ready:
        pytest.skip(reason)


def _blob(items: list[str]) -> str:
    """Join items into a lower-cased blob for resilient substring assertions."""
    return " ".join(items).lower()


def _assert_clean_string_list(items):
    assert isinstance(items, list)
    assert all(isinstance(item, str) and item.strip() for item in items), items


def test_extract_bullets_and_checkboxes():
    text = """
    Notes from meeting:
    - [ ] Set up database
    * implement API extract endpoint
    1. Write tests
    Some narrative sentence.
    """.strip()

    items = extract_action_items(text)
    assert "Set up database" in items
    assert "implement API extract endpoint" in items
    assert "Write tests" in items


def test_extract_llm_bullet_list(ollama_ready):
    """A bullet/numbered/checkbox list is extracted into action items."""
    text = """
    Notes from meeting:
    - [ ] Set up database
    * implement API extract endpoint
    1. Write tests
    Some narrative sentence.
    """.strip()

    items = extract_action_items_llm(text)

    _assert_clean_string_list(items)
    assert items, "expected at least one action item"
    blob = _blob(items)
    assert "database" in blob
    assert "api" in blob
    assert "test" in blob


def test_extract_llm_keyword_prefixed_lines(ollama_ready):
    """TODO:/ACTION:/NEXT: prefixed lines are extracted as action items."""
    text = "TODO: refactor the parser\nACTION: ping the design team\nNEXT: book a room"

    items = extract_action_items_llm(text)

    _assert_clean_string_list(items)
    assert items, "expected at least one action item"
    blob = _blob(items)
    assert "refactor" in blob
    assert "design" in blob
    assert "room" in blob


def test_extract_llm_prose_with_implicit_actions(ollama_ready):
    """The model infers action items from narrative prose."""
    text = "We should probably document the API. Also the deploy script is broken."

    items = extract_action_items_llm(text)

    _assert_clean_string_list(items)
    assert items, "expected at least one action item"
    blob = _blob(items)
    assert "document" in blob
    assert "deploy" in blob


def test_extract_llm_returns_unique_items(ollama_ready):
    """Returned items are unique (case-insensitive)."""
    text = "- Write tests\n- Write tests\n- write tests"

    items = extract_action_items_llm(text)

    _assert_clean_string_list(items)
    lowered = [item.lower() for item in items]
    assert len(lowered) == len(set(lowered))


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_extract_llm_empty_input_returns_empty(text):
    """Empty/whitespace input returns [] without contacting the model."""
    assert extract_action_items_llm(text) == []


def test_extract_llm_non_actionable_text(ollama_ready):
    """Clearly non-actionable text does not crash and returns a clean list."""
    text = "The team had lunch together and chatted about the weather."

    items = extract_action_items_llm(text)

    _assert_clean_string_list(items)
    assert len(items) <= 2
