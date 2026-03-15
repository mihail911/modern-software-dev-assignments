def test_note_validation_empty_title(client):
    """Test that empty title is rejected"""
    payload = {"title": "", "content": "Some content"}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 422
    error = r.json()
    assert "detail" in error


def test_note_validation_empty_content(client):
    """Test that empty content is rejected"""
    payload = {"title": "Valid Title", "content": ""}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 422


def test_note_validation_title_too_long(client):
    """Test that title exceeding max length is rejected"""
    payload = {"title": "x" * 201, "content": "Content"}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 422


def test_note_delete_success(client):
    """Test successful note deletion"""
    payload = {"title": "To Delete", "content": "Will be deleted"}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 201
    note_id = r.json()["id"]

    r = client.delete(f"/notes/{note_id}")
    assert r.status_code == 204

    r = client.get(f"/notes/{note_id}")
    assert r.status_code == 404


def test_note_delete_not_found(client):
    """Test deleting non-existent note returns 404"""
    r = client.delete("/notes/99999")
    assert r.status_code == 404


def test_action_item_validation_empty_description(client):
    """Test that empty description is rejected"""
    payload = {"description": ""}
    r = client.post("/action-items/", json=payload)
    assert r.status_code == 422


def test_action_item_validation_description_too_long(client):
    """Test that description exceeding max length is rejected"""
    payload = {"description": "x" * 501}
    r = client.post("/action-items/", json=payload)
    assert r.status_code == 422


def test_action_item_delete_success(client):
    """Test successful action item deletion"""
    payload = {"description": "To Delete"}
    r = client.post("/action-items/", json=payload)
    assert r.status_code == 201
    item_id = r.json()["id"]

    r = client.delete(f"/action-items/{item_id}")
    assert r.status_code == 204

    r = client.get("/action-items/")
    items = r.json()
    assert not any(item["id"] == item_id for item in items)


def test_action_item_delete_not_found(client):
    """Test deleting non-existent action item returns 404"""
    r = client.delete("/action-items/99999")
    assert r.status_code == 404


def test_note_patch_validation(client):
    """Test that patch validation works"""
    payload = {"title": "Original", "content": "Original content"}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 201
    note_id = r.json()["id"]

    r = client.patch(f"/notes/{note_id}", json={"title": ""})
    assert r.status_code == 422

    r = client.patch(f"/notes/{note_id}", json={"title": "x" * 201})
    assert r.status_code == 422


def test_action_item_patch_validation(client):
    """Test that action item patch validation works"""
    payload = {"description": "Original"}
    r = client.post("/action-items/", json=payload)
    assert r.status_code == 201
    item_id = r.json()["id"]

    r = client.patch(f"/action-items/{item_id}", json={"description": ""})
    assert r.status_code == 422

    r = client.patch(f"/action-items/{item_id}", json={"description": "x" * 501})
    assert r.status_code == 422
