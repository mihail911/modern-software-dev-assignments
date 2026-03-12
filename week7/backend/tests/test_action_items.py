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


# ── Pagination ────────────────────────────────────────────────────────────────

def test_action_items_pagination_two_pages(client):
    """10 action items split into two equal pages of 5 via limit + skip."""
    created_ids = []
    for i in range(1, 11):
        r = client.post("/action-items/", json={"description": f"Task {i:02d}"})
        assert r.status_code == 201
        created_ids.append(r.json()["id"])

    # First page
    r = client.get("/action-items/", params={"limit": 5, "skip": 0, "sort": "id"})
    assert r.status_code == 200
    page1 = r.json()
    assert len(page1) == 5

    # Second page
    r = client.get("/action-items/", params={"limit": 5, "skip": 5, "sort": "id"})
    assert r.status_code == 200
    page2 = r.json()
    assert len(page2) == 5

    # Both pages together must cover exactly all 10 created items
    page1_ids = {item["id"] for item in page1}
    page2_ids = {item["id"] for item in page2}
    assert page1_ids.isdisjoint(page2_ids), "Pages must not overlap"
    assert page1_ids | page2_ids == set(created_ids)


def test_action_items_pagination_limit_exceeds_records(client):
    """Requesting more records than exist returns only the available ones."""
    for i in range(1, 4):
        r = client.post("/action-items/", json={"description": f"Few Task {i}"})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"limit": 50})
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_action_items_pagination_skip_beyond_records(client):
    """Skipping past all available records returns an empty list."""
    for i in range(1, 4):
        r = client.post("/action-items/", json={"description": f"Sparse Task {i}"})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"limit": 10, "skip": 10})
    assert r.status_code == 200
    assert r.json() == []


def test_action_items_pagination_partial_last_page(client):
    """When total records are not divisible by limit, the last page is smaller."""
    for i in range(1, 8):  # 7 items
        r = client.post("/action-items/", json={"description": f"Partial Task {i:02d}"})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"limit": 5, "skip": 0, "sort": "id"})
    assert r.status_code == 200
    assert len(r.json()) == 5

    r = client.get("/action-items/", params={"limit": 5, "skip": 5, "sort": "id"})
    assert r.status_code == 200
    assert len(r.json()) == 2  # only 2 records remain on the last page


# ── Sorting ───────────────────────────────────────────────────────────────────

def test_action_items_sort_ascending_description(client):
    """sort=description returns items in A→Z description order."""
    for desc in ("Zeta task", "Alpha task", "Mu task", "Beta task"):
        r = client.post("/action-items/", json={"description": desc})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"sort": "description"})
    assert r.status_code == 200
    descriptions = [item["description"] for item in r.json()]
    assert descriptions == sorted(descriptions), f"Expected ascending order, got: {descriptions}"


def test_action_items_sort_descending_description(client):
    """sort=-description returns items in Z→A description order."""
    for desc in ("Zeta task", "Alpha task", "Mu task", "Beta task"):
        r = client.post("/action-items/", json={"description": desc})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"sort": "-description"})
    assert r.status_code == 200
    descriptions = [item["description"] for item in r.json()]
    assert descriptions == sorted(descriptions, reverse=True), f"Expected descending order, got: {descriptions}"


def test_action_items_sort_ascending_created_at(client):
    """sort=created_at returns oldest-first order."""
    for i in range(1, 6):
        r = client.post("/action-items/", json={"description": f"Ts Task {i}"})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"sort": "created_at"})
    assert r.status_code == 200
    timestamps = [item["created_at"] for item in r.json()]
    assert timestamps == sorted(timestamps), "Expected oldest-first order"


def test_action_items_sort_descending_created_at(client):
    """sort=-created_at (default) returns newest-first order."""
    for i in range(1, 6):
        r = client.post("/action-items/", json={"description": f"Ts Task {i}"})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"sort": "-created_at"})
    assert r.status_code == 200
    timestamps = [item["created_at"] for item in r.json()]
    assert timestamps == sorted(timestamps, reverse=True), "Expected newest-first order"


def test_action_items_sort_invalid_field_falls_back_to_default(client):
    """An unrecognised sort field silently falls back to -created_at."""
    for i in range(1, 4):
        r = client.post("/action-items/", json={"description": f"Fallback Task {i}"})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"sort": "nonexistent_field"})
    assert r.status_code == 200
    assert len(r.json()) == 3  # all records still returned


# ── Pagination + Filter combined ──────────────────────────────────────────────

def test_action_items_pagination_with_completed_filter(client):
    """Pagination works correctly when filtered to completed=True."""
    ids = []
    for i in range(1, 11):
        r = client.post("/action-items/", json={"description": f"Mixed Task {i:02d}"})
        assert r.status_code == 201
        ids.append(r.json()["id"])

    # Complete the first 6 items
    for item_id in ids[:6]:
        r = client.put(f"/action-items/{item_id}/complete")
        assert r.status_code == 200

    # Page 1 of completed items
    r = client.get("/action-items/", params={"completed": True, "limit": 3, "skip": 0, "sort": "id"})
    assert r.status_code == 200
    page1 = r.json()
    assert len(page1) == 3
    assert all(item["completed"] for item in page1)

    # Page 2 of completed items
    r = client.get("/action-items/", params={"completed": True, "limit": 3, "skip": 3, "sort": "id"})
    assert r.status_code == 200
    page2 = r.json()
    assert len(page2) == 3
    assert all(item["completed"] for item in page2)

    # Together they cover exactly the 6 completed items
    completed_ids = {item["id"] for item in page1 + page2}
    assert completed_ids == set(ids[:6])


def test_action_items_pagination_with_pending_filter(client):
    """Pagination with completed=False returns only pending items."""
    ids = []
    for i in range(1, 11):
        r = client.post("/action-items/", json={"description": f"Pending Task {i:02d}"})
        assert r.status_code == 201
        ids.append(r.json()["id"])

    # Complete only the first 4; leave 6 pending
    for item_id in ids[:4]:
        r = client.put(f"/action-items/{item_id}/complete")
        assert r.status_code == 200

    r = client.get("/action-items/", params={"completed": False, "limit": 10, "skip": 0})
    assert r.status_code == 200
    pending = r.json()
    assert len(pending) == 6
    assert all(not item["completed"] for item in pending)


# ── Pagination + Sort combined ────────────────────────────────────────────────

def test_action_items_pagination_with_sort_preserves_order(client):
    """Items on page 2 must be the direct continuation of page 1 when sorted by description."""
    descriptions = [f"Task {chr(ord('A') + i)}" for i in range(10)]  # Task A … Task J
    for desc in descriptions:
        r = client.post("/action-items/", json={"description": desc})
        assert r.status_code == 201

    r = client.get("/action-items/", params={"sort": "description", "limit": 5, "skip": 0})
    assert r.status_code == 200
    page1_descs = [item["description"] for item in r.json()]

    r = client.get("/action-items/", params={"sort": "description", "limit": 5, "skip": 5})
    assert r.status_code == 200
    page2_descs = [item["description"] for item in r.json()]

    combined = page1_descs + page2_descs
    assert combined == sorted(combined), "Combined pages must be in continuous sorted order"
    assert combined == sorted(descriptions), "All descriptions must be present in sorted order"
