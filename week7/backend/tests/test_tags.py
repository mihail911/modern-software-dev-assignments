def test_create_and_list_tags(client):
    """Test creating and listing tags"""
    payload = {"name": "python"}
    r = client.post("/tags/", json=payload)
    assert r.status_code == 201
    tag = r.json()
    assert tag["name"] == "python"
    assert "id" in tag

    r = client.get("/tags/")
    assert r.status_code == 200
    tags = r.json()
    assert len(tags) >= 1
    assert any(t["name"] == "python" for t in tags)


def test_create_duplicate_tag(client):
    """Test that duplicate tag names are rejected"""
    payload = {"name": "duplicate"}
    r = client.post("/tags/", json=payload)
    assert r.status_code == 201

    r = client.post("/tags/", json=payload)
    assert r.status_code == 400


def test_delete_tag(client):
    """Test deleting a tag"""
    payload = {"name": "todelete"}
    r = client.post("/tags/", json=payload)
    assert r.status_code == 201
    tag_id = r.json()["id"]

    r = client.delete(f"/tags/{tag_id}")
    assert r.status_code == 204

    r = client.get("/tags/")
    tags = r.json()
    assert not any(t["id"] == tag_id for t in tags)


def test_attach_tag_to_note(client):
    """Test attaching a tag to a note"""
    note_payload = {"title": "Test Note", "content": "Content"}
    r = client.post("/notes/", json=note_payload)
    assert r.status_code == 201
    note_id = r.json()["id"]

    tag_payload = {"name": "backend"}
    r = client.post("/tags/", json=tag_payload)
    assert r.status_code == 201
    tag_id = r.json()["id"]

    r = client.post(f"/tags/notes/{note_id}/tags/{tag_id}")
    assert r.status_code == 204

    r = client.get(f"/notes/{note_id}")
    assert r.status_code == 200
    note = r.json()
    assert "tags" in note
    assert len(note["tags"]) == 1
    assert note["tags"][0]["name"] == "backend"


def test_detach_tag_from_note(client):
    """Test detaching a tag from a note"""
    note_payload = {"title": "Test Note", "content": "Content"}
    r = client.post("/notes/", json=note_payload)
    note_id = r.json()["id"]

    tag_payload = {"name": "frontend"}
    r = client.post("/tags/", json=tag_payload)
    tag_id = r.json()["id"]

    r = client.post(f"/tags/notes/{note_id}/tags/{tag_id}")
    assert r.status_code == 204

    r = client.delete(f"/tags/notes/{note_id}/tags/{tag_id}")
    assert r.status_code == 204

    r = client.get(f"/notes/{note_id}")
    note = r.json()
    assert len(note["tags"]) == 0


def test_attach_tag_note_not_found(client):
    """Test attaching tag to non-existent note"""
    tag_payload = {"name": "test"}
    r = client.post("/tags/", json=tag_payload)
    tag_id = r.json()["id"]

    r = client.post(f"/tags/notes/99999/tags/{tag_id}")
    assert r.status_code == 404


def test_attach_tag_tag_not_found(client):
    """Test attaching non-existent tag to note"""
    note_payload = {"title": "Test", "content": "Content"}
    r = client.post("/notes/", json=note_payload)
    note_id = r.json()["id"]

    r = client.post(f"/tags/notes/{note_id}/tags/99999")
    assert r.status_code == 404


def test_multiple_tags_on_note(client):
    """Test attaching multiple tags to a single note"""
    note_payload = {"title": "Multi-tag Note", "content": "Content"}
    r = client.post("/notes/", json=note_payload)
    note_id = r.json()["id"]

    tag1 = client.post("/tags/", json={"name": "python"}).json()
    tag2 = client.post("/tags/", json={"name": "fastapi"}).json()
    tag3 = client.post("/tags/", json={"name": "testing"}).json()

    client.post(f"/tags/notes/{note_id}/tags/{tag1['id']}")
    client.post(f"/tags/notes/{note_id}/tags/{tag2['id']}")
    client.post(f"/tags/notes/{note_id}/tags/{tag3['id']}")

    r = client.get(f"/notes/{note_id}")
    note = r.json()
    assert len(note["tags"]) == 3
    tag_names = [t["name"] for t in note["tags"]]
    assert "python" in tag_names
    assert "fastapi" in tag_names
    assert "testing" in tag_names
