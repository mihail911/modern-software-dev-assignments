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


# ── Pagination ────────────────────────────────────────────────────────────────

def test_notes_pagination_two_pages(client):
    """10 notes split into two equal pages of 5 via limit + skip."""
    created_ids = []
    for i in range(1, 11):
        r = client.post("/notes/", json={"title": f"Note {i:02d}", "content": "Pagination content"})
        assert r.status_code == 201
        created_ids.append(r.json()["id"])

    # First page
    r = client.get("/notes/", params={"limit": 5, "skip": 0, "sort": "id"})
    assert r.status_code == 200
    page1 = r.json()
    assert len(page1) == 5

    # Second page
    r = client.get("/notes/", params={"limit": 5, "skip": 5, "sort": "id"})
    assert r.status_code == 200
    page2 = r.json()
    assert len(page2) == 5

    # Both pages together must cover exactly all 10 created notes
    page1_ids = {item["id"] for item in page1}
    page2_ids = {item["id"] for item in page2}
    assert page1_ids.isdisjoint(page2_ids), "Pages must not overlap"
    assert page1_ids | page2_ids == set(created_ids)


def test_notes_pagination_limit_exceeds_records(client):
    """Requesting more records than exist returns only the available ones."""
    for i in range(1, 4):
        r = client.post("/notes/", json={"title": f"Few Note {i}", "content": "Only a few notes"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"limit": 50})
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_notes_pagination_skip_beyond_records(client):
    """Skipping past all available records returns an empty list."""
    for i in range(1, 4):
        r = client.post("/notes/", json={"title": f"Sparse Note {i}", "content": "Sparse content"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"limit": 10, "skip": 10})
    assert r.status_code == 200
    assert r.json() == []


def test_notes_pagination_partial_last_page(client):
    """When total records are not divisible by limit, the last page is smaller."""
    for i in range(1, 8):  # 7 notes
        r = client.post("/notes/", json={"title": f"Partial Note {i:02d}", "content": "Partial page"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"limit": 5, "skip": 0, "sort": "id"})
    assert r.status_code == 200
    assert len(r.json()) == 5

    r = client.get("/notes/", params={"limit": 5, "skip": 5, "sort": "id"})
    assert r.status_code == 200
    assert len(r.json()) == 2  # only 2 records remain on the last page


# ── Sorting ───────────────────────────────────────────────────────────────────

def test_notes_sort_ascending_title(client):
    """sort=title returns notes in A→Z title order."""
    for title in ("Mango", "Apple", "Zebra", "Banana"):
        r = client.post("/notes/", json={"title": title, "content": "Sort test content"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"sort": "title"})
    assert r.status_code == 200
    titles = [item["title"] for item in r.json()]
    assert titles == sorted(titles), f"Expected ascending order, got: {titles}"


def test_notes_sort_descending_title(client):
    """sort=-title returns notes in Z→A title order."""
    for title in ("Mango", "Apple", "Zebra", "Banana"):
        r = client.post("/notes/", json={"title": title, "content": "Sort test content"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"sort": "-title"})
    assert r.status_code == 200
    titles = [item["title"] for item in r.json()]
    assert titles == sorted(titles, reverse=True), f"Expected descending order, got: {titles}"


def test_notes_sort_ascending_created_at(client):
    """sort=created_at returns oldest-first order."""
    for i in range(1, 6):
        r = client.post("/notes/", json={"title": f"Ts Note {i}", "content": "Timestamp sort"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"sort": "created_at"})
    assert r.status_code == 200
    timestamps = [item["created_at"] for item in r.json()]
    assert timestamps == sorted(timestamps), "Expected oldest-first order"


def test_notes_sort_descending_created_at(client):
    """sort=-created_at (default) returns newest-first order."""
    for i in range(1, 6):
        r = client.post("/notes/", json={"title": f"Ts Note {i}", "content": "Timestamp sort"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"sort": "-created_at"})
    assert r.status_code == 200
    timestamps = [item["created_at"] for item in r.json()]
    assert timestamps == sorted(timestamps, reverse=True), "Expected newest-first order"


def test_notes_sort_invalid_field_falls_back_to_default(client):
    """An unrecognised sort field silently falls back to -created_at."""
    for i in range(1, 4):
        r = client.post("/notes/", json={"title": f"Fallback Note {i}", "content": "Fallback sort"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"sort": "nonexistent_field"})
    assert r.status_code == 200
    assert len(r.json()) == 3  # all records still returned


# ── Pagination + Sort combined ────────────────────────────────────────────────

def test_notes_pagination_with_sort_preserves_order(client):
    """Items on page 2 must be the direct continuation of page 1 when sorted by title."""
    titles = [f"Note {chr(ord('A') + i)}" for i in range(10)]  # Note A … Note J
    for title in titles:
        r = client.post("/notes/", json={"title": title, "content": "Combined test"})
        assert r.status_code == 201

    r = client.get("/notes/", params={"sort": "title", "limit": 5, "skip": 0})
    assert r.status_code == 200
    page1_titles = [item["title"] for item in r.json()]

    r = client.get("/notes/", params={"sort": "title", "limit": 5, "skip": 5})
    assert r.status_code == 200
    page2_titles = [item["title"] for item in r.json()]

    combined = page1_titles + page2_titles
    assert combined == sorted(combined), "Combined pages must be in continuous sorted order"
    assert combined == sorted(titles), "All titles must be present in sorted order"
