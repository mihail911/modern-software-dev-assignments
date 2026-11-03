def test_note_tag_many_to_many(client):
    r = client.post("/notes/", json={"title": "Tagged", "content": "Hello", "tags": ["Work", "urgent", "work"]})
    assert r.status_code == 201, r.text
    note = r.json()
    tag_names = {t["name"] for t in note["tags"]}
    assert tag_names == {"work", "urgent"}

    r = client.get("/notes/", params={"tag": "work"})
    assert r.status_code == 200
    titles = [n["title"] for n in r.json()]
    assert "Tagged" in titles

    r = client.get("/tags/")
    assert r.status_code == 200
    listed = {t["name"] for t in r.json()}
    assert {"work", "urgent"} <= listed

    tag_id = next(t["id"] for t in note["tags"] if t["name"] == "work")
    r = client.get(f"/tags/{tag_id}/notes")
    assert r.status_code == 200
    assert any(n["id"] == note["id"] for n in r.json())

    r = client.post(f"/notes/{note['id']}/tags", json={"name": "personal"})
    assert r.status_code == 200
    assert {t["name"] for t in r.json()["tags"]} == {"work", "urgent", "personal"}

    personal_id = next(t["id"] for t in r.json()["tags"] if t["name"] == "personal")
    r = client.delete(f"/notes/{note['id']}/tags/{personal_id}")
    assert r.status_code == 200
    assert {t["name"] for t in r.json()["tags"]} == {"work", "urgent"}

    r = client.patch(f"/notes/{note['id']}", json={"tags": ["focus"]})
    assert r.status_code == 200
    assert [t["name"] for t in r.json()["tags"]] == ["focus"]
