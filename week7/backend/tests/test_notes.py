def test_create_list_and_patch_notes(client):
    payload = {"title": "Test", "content": "Hello world"}
    r = client.post("/notes/", json=payload)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["title"] == "Test"
    assert "created_at" in data and "updated_at" in data

    r = client.get("/notes/")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1

    r = client.get("/notes/", params={"q": "Hello", "limit": 10, "sort": "-created_at"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1

    note_id = data["id"]
    r = client.patch(f"/notes/{note_id}", json={"title": "Updated"})
    assert r.status_code == 200
    patched = r.json()
    assert patched["title"] == "Updated"


def test_note_validation_and_delete(client):
    r = client.post("/notes/", json={"title": " ", "content": "Body"})
    assert r.status_code == 422

    created = client.post("/notes/", json={"title": "Delete me", "content": "Body"}).json()

    r = client.patch(f"/notes/{created['id']}", json={})
    assert r.status_code == 422

    r = client.delete(f"/notes/{created['id']}")
    assert r.status_code == 204

    r = client.get(f"/notes/{created['id']}")
    assert r.status_code == 404


