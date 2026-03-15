def test_notes_pagination_basic(client):
    """Test basic pagination for notes"""
    for i in range(15):
        client.post("/notes/", json={"title": f"Note {i}", "content": f"Content {i}"})

    r = client.get("/notes/", params={"skip": 0, "limit": 5})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 5

    r = client.get("/notes/", params={"skip": 5, "limit": 5})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 5

    r = client.get("/notes/", params={"skip": 10, "limit": 5})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 5


def test_notes_pagination_empty_page(client):
    """Test pagination with skip beyond available items"""
    client.post("/notes/", json={"title": "Only Note", "content": "Content"})

    r = client.get("/notes/", params={"skip": 100, "limit": 10})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 0


def test_notes_pagination_limit_exceeds_max(client):
    """Test that limit is capped at maximum"""
    for i in range(10):
        client.post("/notes/", json={"title": f"Note {i}", "content": f"Content {i}"})

    r = client.get("/notes/", params={"limit": 300})
    assert r.status_code == 422


def test_notes_sorting_by_created_at_desc(client):
    """Test sorting notes by created_at descending"""
    note1 = client.post("/notes/", json={"title": "First", "content": "Content"}).json()
    note2 = client.post("/notes/", json={"title": "Second", "content": "Content"}).json()
    note3 = client.post("/notes/", json={"title": "Third", "content": "Content"}).json()

    r = client.get("/notes/", params={"sort": "-created_at"})
    assert r.status_code == 200
    notes = r.json()

    ids = [n["id"] for n in notes]
    assert ids.index(note3["id"]) < ids.index(note2["id"])
    assert ids.index(note2["id"]) < ids.index(note1["id"])


def test_notes_sorting_by_created_at_asc(client):
    """Test sorting notes by created_at ascending"""
    note1 = client.post("/notes/", json={"title": "First", "content": "Content"}).json()
    note2 = client.post("/notes/", json={"title": "Second", "content": "Content"}).json()
    note3 = client.post("/notes/", json={"title": "Third", "content": "Content"}).json()

    r = client.get("/notes/", params={"sort": "created_at"})
    assert r.status_code == 200
    notes = r.json()

    ids = [n["id"] for n in notes]
    assert ids.index(note1["id"]) < ids.index(note2["id"])
    assert ids.index(note2["id"]) < ids.index(note3["id"])


def test_notes_sorting_by_title(client):
    """Test sorting notes by title"""
    client.post("/notes/", json={"title": "Zebra", "content": "Content"})
    client.post("/notes/", json={"title": "Apple", "content": "Content"})
    client.post("/notes/", json={"title": "Mango", "content": "Content"})

    r = client.get("/notes/", params={"sort": "title"})
    assert r.status_code == 200
    notes = r.json()
    titles = [n["title"] for n in notes]

    assert titles.index("Apple") < titles.index("Mango")
    assert titles.index("Mango") < titles.index("Zebra")


def test_notes_sorting_invalid_field(client):
    """Test sorting with invalid field falls back to default"""
    client.post("/notes/", json={"title": "Note", "content": "Content"})

    r = client.get("/notes/", params={"sort": "invalid_field"})
    assert r.status_code == 200


def test_action_items_pagination_basic(client):
    """Test basic pagination for action items"""
    for i in range(12):
        client.post("/action-items/", json={"description": f"Task {i}"})

    r = client.get("/action-items/", params={"skip": 0, "limit": 5})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 5

    r = client.get("/action-items/", params={"skip": 5, "limit": 5})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 5


def test_action_items_sorting_by_created_at(client):
    """Test sorting action items by created_at"""
    item1 = client.post("/action-items/", json={"description": "First"}).json()
    item2 = client.post("/action-items/", json={"description": "Second"}).json()
    item3 = client.post("/action-items/", json={"description": "Third"}).json()

    r = client.get("/action-items/", params={"sort": "-created_at"})
    assert r.status_code == 200
    items = r.json()

    ids = [i["id"] for i in items]
    assert ids.index(item3["id"]) < ids.index(item2["id"])
    assert ids.index(item2["id"]) < ids.index(item1["id"])


def test_action_items_filter_completed_with_pagination(client):
    """Test filtering completed items with pagination"""
    for i in range(10):
        item = client.post("/action-items/", json={"description": f"Task {i}"}).json()
        if i % 2 == 0:
            client.put(f"/action-items/{item['id']}/complete")

    r = client.get("/action-items/", params={"completed": True, "limit": 3})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 3
    assert all(item["completed"] for item in items)

    r = client.get("/action-items/", params={"completed": False, "limit": 3})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 3
    assert all(not item["completed"] for item in items)


def test_notes_search_with_pagination(client):
    """Test search with pagination"""
    for i in range(15):
        client.post("/notes/", json={"title": f"Python Tutorial {i}", "content": "Learn Python"})

    r = client.get("/notes/", params={"q": "Python", "skip": 0, "limit": 5})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 5

    r = client.get("/notes/", params={"q": "Python", "skip": 10, "limit": 5})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 5


def test_notes_search_with_sorting(client):
    """Test search with sorting"""
    client.post("/notes/", json={"title": "Zebra Python", "content": "Content"})
    client.post("/notes/", json={"title": "Apple Python", "content": "Content"})
    client.post("/notes/", json={"title": "Mango Python", "content": "Content"})

    r = client.get("/notes/", params={"q": "Python", "sort": "title"})
    assert r.status_code == 200
    notes = r.json()
    titles = [n["title"] for n in notes]

    assert titles.index("Apple Python") < titles.index("Mango Python")
    assert titles.index("Mango Python") < titles.index("Zebra Python")


def test_pagination_with_zero_limit(client):
    """Test pagination with zero limit returns empty list"""
    client.post("/notes/", json={"title": "Note", "content": "Content"})

    r = client.get("/notes/", params={"limit": 0})
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 0


def test_pagination_with_negative_skip(client):
    """Test pagination with negative skip returns all items"""
    client.post("/notes/", json={"title": "Note", "content": "Content"})

    r = client.get("/notes/", params={"skip": -1})
    assert r.status_code == 200
