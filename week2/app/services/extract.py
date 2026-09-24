"""Action-item extraction.

Two extractors are provided:

* :func:`extract_action_items` - deterministic, heuristic/rule-based.
* :func:`extract_action_items_llm` - LLM-powered via Ollama structured outputs.

Both share the same contract: given free-form text they return a de-duplicated
list of action-item strings, preserving order.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dotenv import load_dotenv
from ollama import chat

from ..config import get_settings
from ..errors import UpstreamServiceError
from ..schemas import ExtractedItems

load_dotenv()

BULLET_PREFIX_PATTERN = re.compile(r"^\s*([-*•]|\d+\.)\s+")
KEYWORD_PREFIXES = (
    "todo:",
    "action:",
    "next:",
)

# Model used for the LLM-powered extractor. Sourced from the central settings so
# it can be overridden via the OLLAMA_MODEL environment variable.
OLLAMA_MODEL = get_settings().ollama_model

# JSON schema passed to Ollama's structured-outputs API. It is derived from the
# ``ExtractedItems`` pydantic model, keeping the wire contract and the response
# validation in sync.
ACTION_ITEMS_SCHEMA: dict[str, Any] = ExtractedItems.model_json_schema()

LLM_SYSTEM_PROMPT = """
You extract action items from notes.

An action item is anything that represents a task someone should do.

Rules:
1. Every bullet point is an action item.
2. Every line beginning with TODO:, ACTION:, or NEXT: is an action item.
3. Every checkbox item such as [ ] or [TODO] is an action item.
4. Imperative statements such as "fix the bug", "update the README", or "check the logs" are action items.
5. Do not include questions, observations, explanations, or background information unless they clearly describe a task to perform.
6. Remove bullet markers, checkbox markers, and TODO/ACTION/NEXT prefixes.
7. Preserve the original meaning and wording as much as possible.
8. Return an empty list only when there are genuinely no action items.

Return JSON matching the provided schema.
"""


def _is_action_line(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return False
    if BULLET_PREFIX_PATTERN.match(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in KEYWORD_PREFIXES):
        return True
    if "[ ]" in stripped or "[todo]" in stripped:
        return True
    return False


def extract_action_items(text: str) -> list[str]:
    """Extract action items using deterministic line-prefix heuristics."""
    lines = text.splitlines()
    extracted: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_action_line(line):
            cleaned = BULLET_PREFIX_PATTERN.sub("", line)
            cleaned = cleaned.strip()
            # Trim common checkbox markers
            cleaned = cleaned.removeprefix("[ ]").strip()
            cleaned = cleaned.removeprefix("[todo]").strip()
            extracted.append(cleaned)
    # Fallback: if nothing matched, heuristically split into sentences and pick
    # imperative-like ones.
    if not extracted:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            if _looks_imperative(s):
                extracted.append(s)
    # Deduplicate while preserving order
    return _deduplicate(extracted)


def extract_action_items_llm(text: str) -> list[str]:
    """Extract action items from ``text`` using an Ollama-hosted LLM.

    This is the LLM-powered counterpart to :func:`extract_action_items`. It asks
    the model for a structured JSON response (an object containing an ``items``
    array of strings) rather than relying on line-prefix heuristics.

    Args:
        text: Free-form notes to extract action items from.

    Returns:
        A de-duplicated list of action item strings, preserving model order.
        Returns an empty list for empty/whitespace-only input.

    Raises:
        UpstreamServiceError: If the model request fails or the response cannot
            be parsed as JSON containing an ``items`` array.
    """
    if not text or not text.strip():
        return []

    settings = get_settings()
    prompt = f"Extract all action items from this text: \n\n{text}"

    try:
        response = chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format=ACTION_ITEMS_SCHEMA,
            options={"temperature": settings.llm_temperature},
        )
    except Exception as exc:  # noqa: BLE001 - normalise any client failure
        raise UpstreamServiceError("Failed to reach the LLM service", details=str(exc)) from exc

    items = _parse_action_items(response.message.content)
    return _deduplicate(items)


def _parse_action_items(content: Any) -> list[str]:
    """Normalize the model's structured output into a list of strings."""
    # Newer ollama clients may already hand back a parsed mapping; otherwise the
    # content is a JSON string (possibly wrapped in markdown fences).
    if isinstance(content, dict):
        payload: Any = content
    else:
        raw = (content or "").strip()
        if not raw:
            return []
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UpstreamServiceError("Model did not return valid JSON", details=str(exc)) from exc

    try:
        parsed = ExtractedItems.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface as an upstream contract error
        raise UpstreamServiceError(
            "Model response did not match the expected schema", details=str(exc)
        ) from exc

    return [item.strip() for item in parsed.items if item.strip()]


def _deduplicate(items: list[str]) -> list[str]:
    """Deduplicate while preserving order, case-insensitively."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(item)
    return unique


def _looks_imperative(sentence: str) -> bool:
    words = re.findall(r"[A-Za-z']+", sentence)
    if not words:
        return False
    first = words[0]
    # Crude heuristic: treat these as imperative starters
    imperative_starters = {
        "add",
        "create",
        "implement",
        "fix",
        "update",
        "write",
        "check",
        "verify",
        "refactor",
        "document",
        "design",
        "investigate",
    }
    return first.lower() in imperative_starters
