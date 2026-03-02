import os
import pytest
from unittest.mock import patch, MagicMock
import json

from ..app.services.extract import extract_action_items, extract_action_items_llm


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


@patch("app.services.extract.chat")
def test_extract_action_items_llm_with_bullet_list(mock_chat):
    """Test extract_action_items_llm with normal bullet list todos."""
    text = """
    Project tasks:
    - Set up database connection
    * Implement REST API endpoint
    • Write comprehensive tests
    """
    
    # Mock the ollama.chat response
    mock_response = MagicMock()
    mock_response.message.content = json.dumps({
        "action_items": [
            "Set up database connection",
            "Implement REST API endpoint",
            "Write comprehensive tests"
        ]
    })
    mock_chat.return_value = mock_response
    
    items = extract_action_items_llm(text)
    
    assert len(items) == 3
    assert "Set up database connection" in items
    assert "Implement REST API endpoint" in items
    assert "Write comprehensive tests" in items
    mock_chat.assert_called_once()


@patch("app.services.extract.chat")
def test_extract_action_items_llm_with_keyword_prefixes(mock_chat):
    """Test extract_action_items_llm with keyword-prefixed lines."""
    text = """
    Meeting notes:
    todo: Review pull requests
    action: Schedule client meeting
    next: Deploy to production
    """
    
    # Mock the ollama.chat response
    mock_response = MagicMock()
    mock_response.message.content = json.dumps({
        "action_items": [
            "Review pull requests",
            "Schedule client meeting",
            "Deploy to production"
        ]
    })
    mock_chat.return_value = mock_response
    
    items = extract_action_items_llm(text)
    
    assert len(items) == 3
    assert "Review pull requests" in items
    assert "Schedule client meeting" in items
    assert "Deploy to production" in items
    mock_chat.assert_called_once()


@patch("app.services.extract.chat")
def test_extract_action_items_llm_empty_input(mock_chat):
    """Test extract_action_items_llm with empty input."""
    text = ""
    
    # Mock the ollama.chat response
    mock_response = MagicMock()
    mock_response.message.content = json.dumps({
        "action_items": []
    })
    mock_chat.return_value = mock_response
    
    items = extract_action_items_llm(text)
    
    assert len(items) == 0
    assert items == []
    mock_chat.assert_called_once()
