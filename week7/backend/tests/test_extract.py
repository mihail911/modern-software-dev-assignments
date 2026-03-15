from backend.app.services.extract import extract_action_items, extract_all, extract_hashtags


def test_extract_action_items():
    text = """
    This is a note
    - TODO: write tests
    - ACTION: review PR
    - Ship it!
    Not actionable
    """.strip()
    items = extract_action_items(text)
    assert "TODO: write tests" in items
    assert "ACTION: review PR" in items
    assert "Ship it!" in items


def test_extract_markdown_checkboxes():
    text = """
    - [ ] Implement feature A
    - [x] Complete feature B
    - [ ] Write documentation
    Regular line
    """
    items = extract_action_items(text)
    assert "Implement feature A" in items
    assert "Write documentation" in items
    assert len([i for i in items if "Complete feature B" in i]) == 0


def test_extract_hashtags():
    text = """
    This is a note about #python and #fastapi
    We should also consider #testing and #ci-cd
    No hashtag here
    """
    tags = extract_hashtags(text)
    assert "python" in tags
    assert "fastapi" in tags
    assert "testing" in tags


def test_extract_hashtags_deduplication():
    text = "#python #python #fastapi #python"
    tags = extract_hashtags(text)
    assert len(tags) == 2
    assert "python" in tags
    assert "fastapi" in tags


def test_extract_all():
    text = """
    Project notes #backend #api
    - [ ] Add authentication
    - TODO: Write tests
    Deploy it!
    """
    result = extract_all(text)
    assert "action_items" in result
    assert "hashtags" in result
    assert len(result["action_items"]) == 3
    assert "backend" in result["hashtags"]
    assert "api" in result["hashtags"]


def test_extract_endpoint(client):
    payload = {
        "title": "Project Tasks",
        "content": """
        #backend #api
        - [ ] Implement auth
        - TODO: Add tests
        Ship it!
        """,
    }
    r = client.post("/notes/", json=payload)
    assert r.status_code == 201
    note_id = r.json()["id"]

    r = client.post(f"/notes/{note_id}/extract")
    assert r.status_code == 200
    data = r.json()
    assert "extracted" in data
    assert "action_items" in data["extracted"]
    assert "hashtags" in data["extracted"]
    assert len(data["extracted"]["action_items"]) == 3
    assert "backend" in data["extracted"]["hashtags"]


def test_extract_endpoint_not_found(client):
    r = client.post("/notes/99999/extract")
    assert r.status_code == 404
