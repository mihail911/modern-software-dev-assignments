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
    assert patched["content"] == "Hello world"


def test_delete_note_success_and_missing(client):
    r = client.post("/notes/", json={"title": "To delete", "content": "gone soon"})
    assert r.status_code == 201
    note_id = r.json()["id"]

    r = client.delete(f"/notes/{note_id}")
    assert r.status_code == 204
    assert r.content == b""

    r = client.get(f"/notes/{note_id}")
    assert r.status_code == 404

    r = client.delete(f"/notes/{note_id}")
    assert r.status_code == 404


def test_note_rejects_blank_and_whitespace_inputs(client):
    r = client.post("/notes/", json={"title": "", "content": "ok"})
    assert r.status_code == 422

    r = client.post("/notes/", json={"title": "   ", "content": "ok"})
    assert r.status_code == 422

    r = client.post("/notes/", json={"title": "ok", "content": "   "})
    assert r.status_code == 422

    created = client.post("/notes/", json={"title": "Keep", "content": "Original"})
    assert created.status_code == 201
    note_id = created.json()["id"]

    r = client.patch(f"/notes/{note_id}", json={"title": "   "})
    assert r.status_code == 422

    r = client.patch(f"/notes/{note_id}", json={"content": ""})
    assert r.status_code == 422

    r = client.patch(f"/notes/{note_id}", json={"title": "Still Keep"})
    assert r.status_code == 200
    patched = r.json()
    assert patched["title"] == "Still Keep"
    assert patched["content"] == "Original"


def test_notes_pagination_and_sort_validation(client):
    client.post("/notes/", json={"title": "Alpha", "content": "a"})
    client.post("/notes/", json={"title": "Beta", "content": "b"})

    r = client.get("/notes/", params={"skip": -1})
    assert r.status_code == 422

    r = client.get("/notes/", params={"limit": 0})
    assert r.status_code == 422

    r = client.get("/notes/", params={"limit": 201})
    assert r.status_code == 422

    r = client.get("/notes/", params={"sort": "not_a_column"})
    assert r.status_code == 422

    r = client.get("/notes/", params={"sort": "__table__"})
    assert r.status_code == 422

    r = client.get("/notes/", params={"sort": "title", "limit": 50, "skip": 0})
    assert r.status_code == 200
    titles = [item["title"] for item in r.json()]
    assert titles == sorted(titles)
