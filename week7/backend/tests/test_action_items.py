def test_create_complete_list_and_patch_action_item(client):
    payload = {"description": "Ship it"}
    r = client.post("/action-items/", json=payload)
    assert r.status_code == 201, r.text
    item = r.json()
    assert item["completed"] is False
    assert "created_at" in item and "updated_at" in item

    r = client.put(f"/action-items/{item['id']}/complete")
    assert r.status_code == 200
    done = r.json()
    assert done["completed"] is True

    r = client.get("/action-items/", params={"completed": True, "limit": 5, "sort": "-created_at"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1

    r = client.patch(f"/action-items/{item['id']}", json={"description": "Updated"})
    assert r.status_code == 200
    patched = r.json()
    assert patched["description"] == "Updated"
    assert patched["completed"] is True


def test_delete_action_item_success_and_missing(client):
    r = client.post("/action-items/", json={"description": "Remove me"})
    assert r.status_code == 201
    item_id = r.json()["id"]

    r = client.delete(f"/action-items/{item_id}")
    assert r.status_code == 204
    assert r.content == b""

    r = client.delete(f"/action-items/{item_id}")
    assert r.status_code == 404


def test_action_item_rejects_blank_and_whitespace_inputs(client):
    r = client.post("/action-items/", json={"description": ""})
    assert r.status_code == 422

    r = client.post("/action-items/", json={"description": "   "})
    assert r.status_code == 422

    created = client.post("/action-items/", json={"description": "Keep me"})
    assert created.status_code == 201
    item_id = created.json()["id"]

    r = client.patch(f"/action-items/{item_id}", json={"description": "   "})
    assert r.status_code == 422

    r = client.patch(f"/action-items/{item_id}", json={"completed": True})
    assert r.status_code == 200
    patched = r.json()
    assert patched["description"] == "Keep me"
    assert patched["completed"] is True


def test_action_items_pagination_and_sort_validation(client):
    client.post("/action-items/", json={"description": "Alpha item"})
    client.post("/action-items/", json={"description": "Beta item"})

    r = client.get("/action-items/", params={"skip": -1})
    assert r.status_code == 422

    r = client.get("/action-items/", params={"limit": 0})
    assert r.status_code == 422

    r = client.get("/action-items/", params={"limit": 201})
    assert r.status_code == 422

    r = client.get("/action-items/", params={"sort": "priority"})
    assert r.status_code == 422

    r = client.get("/action-items/", params={"sort": "description", "limit": 50, "skip": 0})
    assert r.status_code == 200
    descriptions = [item["description"] for item in r.json()]
    assert descriptions == sorted(descriptions)
