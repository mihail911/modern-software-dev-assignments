def test_create_and_list_notes(client):
    payload = {"title": "Test", "content": "Hello world"}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["title"] == "Test"

    r = client.get("/notes/")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1

    r = client.get("/notes/search/")
    assert r.status_code == 200

    r = client.get("/notes/search/", params={"q": "Hello"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1


def test_update_note(client):
    r = client.post("/notes/", json={"title": "Original", "content": "Body"})
    note_id = r.json()["id"]

    r = client.put(f"/notes/{note_id}", json={"title": "Updated"})
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "Updated"
    assert data["content"] == "Body"  # unchanged

    r = client.put(f"/notes/{note_id}", json={"content": "New body"})
    assert r.status_code == 200
    assert r.json()["content"] == "New body"
    assert r.json()["title"] == "Updated"  # unchanged


def test_update_note_not_found(client):
    r = client.put("/notes/9999", json={"title": "X"})
    assert r.status_code == 404


def test_update_note_empty_body(client):
    r = client.post("/notes/", json={"title": "T", "content": "C"})
    note_id = r.json()["id"]
    r = client.put(f"/notes/{note_id}", json={})
    assert r.status_code == 422


def test_delete_note(client):
    r = client.post("/notes/", json={"title": "ToDelete", "content": "Gone"})
    note_id = r.json()["id"]

    r = client.delete(f"/notes/{note_id}")
    assert r.status_code == 204

    r = client.get(f"/notes/{note_id}")
    assert r.status_code == 404


def test_delete_note_not_found(client):
    r = client.delete("/notes/9999")
    assert r.status_code == 404


def test_create_note_validation(client):
    # empty title
    r = client.post("/notes/", json={"title": "", "content": "C"})
    assert r.status_code == 422

    # empty content
    r = client.post("/notes/", json={"title": "T", "content": ""})
    assert r.status_code == 422

    # title too long (>200)
    r = client.post("/notes/", json={"title": "x" * 201, "content": "C"})
    assert r.status_code == 422


def test_update_note_validation(client):
    r = client.post("/notes/", json={"title": "T", "content": "C"})
    note_id = r.json()["id"]

    # empty title
    r = client.put(f"/notes/{note_id}", json={"title": ""})
    assert r.status_code == 422

    # title too long
    r = client.put(f"/notes/{note_id}", json={"title": "x" * 201})
    assert r.status_code == 422
