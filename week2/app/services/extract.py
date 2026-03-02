from __future__ import annotations

import os
import re
from typing import List
import json
import logging
from typing import Any
from ollama import chat
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ActionItemList(BaseModel):
    """Schema for structured output of action items."""
    action_items: list[str]

BULLET_PREFIX_PATTERN = re.compile(r"^\s*([-*•]|\d+\.)\s+")
KEYWORD_PREFIXES = (
    "todo:",
    "action:",
    "next:",
)


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


def extract_action_items(text: str) -> List[str]:
    lines = text.splitlines()
    extracted: List[str] = []
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
    # Fallback: if nothing matched, heuristically split into sentences and pick imperative-like ones
    if not extracted:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            if _looks_imperative(s):
                extracted.append(s)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for item in extracted:
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


def extract_action_items_llm(text: str) -> List[str]:
    """Extract action items from text using Ollama LLM with structured output.
    
    Uses ollama3.1:8b model with JSON schema to extract actionable items
    from the provided text.
    
    Args:
        text: The text to extract action items from
        
    Returns:
        A list of extracted action items as strings
    """
    prompt = f"""Extract action items from the following text. Focus on clear, actionable tasks.

Text:
{text}

Return only the action items as a JSON list. Each item should be a concise, actionable task starting with a verb (e.g., "Review", "Update", "Schedule", "Fix"). Remove bullet points or checkboxes from the items."""

    try:
        response = chat(
            model="ollama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            format=ActionItemList.model_json_schema(),
            options={"temperature": 0},
        )
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        # Fall back to heuristic extraction on LLM failure
        return extract_action_items(text)

    # Parse structured response safely
    content = None
    try:
        # response.message.content is expected to be a JSON string
        content = response.message.content
        result = ActionItemList.model_validate_json(content)
        return result.action_items
    except Exception as e:
        logger.warning("Structured parsing failed, attempting raw JSON load: %s", e)
        try:
            parsed = json.loads(content)
            items = parsed.get("action_items") if isinstance(parsed, dict) else None
            if isinstance(items, list):
                # ensure all items are strings
                return [str(i) for i in items]
        except Exception as e2:
            logger.exception("Raw JSON load also failed: %s", e2)

    # Final fallback: use heuristic extractor
    return extract_action_items(text)
