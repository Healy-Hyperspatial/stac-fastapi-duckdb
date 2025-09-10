import pytest


@pytest.mark.asyncio
async def test_get_sort_by_id_asc_desc(app_client):
    # Ascending by id
    resp = await app_client.get("/search?limit=5&sortby=+id")
    assert resp.status_code == 200
    data = resp.json()
    items = data.get("features", [])
    if len(items) > 1:
        ids = [it.get("id") for it in items]
        assert ids == sorted(ids)

    # Descending by id
    resp = await app_client.get("/search?limit=5&sortby=-id")
    assert resp.status_code == 200
    data = resp.json()
    items = data.get("features", [])
    if len(items) > 1:
        ids = [it.get("id") for it in items]
        assert ids == sorted(ids, reverse=True)


@pytest.mark.asyncio
async def test_post_sort_by_id_asc_desc(app_client):
    # Ascending by id
    resp = await app_client.post(
        "/search",
        json={
            "limit": 5,
            "collections": ["io-lulc-9-class"],
            "sortby": [{"field": "id", "direction": "asc"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    items = data.get("features", [])
    if len(items) > 1:
        ids = [it.get("id") for it in items]
        assert ids == sorted(ids)

    # Descending by id
    resp = await app_client.post(
        "/search",
        json={
            "limit": 5,
            "collections": ["io-lulc-9-class"],
            "sortby": [{"field": "id", "direction": "desc"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    items = data.get("features", [])
    if len(items) > 1:
        ids = [it.get("id") for it in items]
        assert ids == sorted(ids, reverse=True)
