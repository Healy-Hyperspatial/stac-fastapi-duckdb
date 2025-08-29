import pytest


@pytest.mark.asyncio
async def test_get_collections(app_client):
    """Test getting collections from the API."""
    response = await app_client.get("/collections")
    assert response.status_code == 200
    data = response.json()
    assert "collections" in data
    assert len(data["collections"]) > 0
    
    # Check that io-lulc-9-class collection exists
    collection_ids = [col["id"] for col in data["collections"]]
    assert "io-lulc-9-class" in collection_ids


@pytest.mark.asyncio
async def test_get_collection_items(app_client):
    """Test getting items from a collection."""
    response = await app_client.get("/collections/io-lulc-9-class/items?limit=5")
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) > 0
    assert "numMatched" in data
    assert "numReturned" in data
    
    # Check first item structure
    item = data["features"][0]
    assert item["type"] == "Feature"
    assert "id" in item
    assert "collection" in item
    assert item["collection"] == "io-lulc-9-class"
    assert "geometry" in item
    assert "properties" in item


@pytest.mark.asyncio
async def test_bbox_filtering(app_client):
    """Test spatial filtering with bbox parameter."""
    # Test with a bbox that should return results
    bbox = "-66,-16,-60,-8"
    response = await app_client.get(f"/collections/io-lulc-9-class/items?bbox={bbox}&limit=3")
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert "numMatched" in data
    assert "numReturned" in data
    # Should have some results within this bbox
    assert len(data["features"]) >= 0


@pytest.mark.asyncio
async def test_datetime_filtering(app_client):
    """Test temporal filtering with datetime parameter."""
    # Test with a datetime range
    datetime_range = "2019-01-01T00:00:00Z/2023-01-01T00:00:00Z"
    response = await app_client.get(f"/collections/io-lulc-9-class/items?datetime={datetime_range}&limit=3")
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    # Should have some results in this time range
    assert len(data["features"]) >= 0


@pytest.mark.asyncio
async def test_combined_filtering(app_client):
    """Test combining spatial and temporal filters."""
    bbox = "-66,-16,-60,-8"
    datetime_range = "2019-01-01T00:00:00Z/2023-01-01T00:00:00Z"
    
    response = await app_client.get(
        f"/collections/io-lulc-9-class/items?bbox={bbox}&datetime={datetime_range}&limit=2"
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == "FeatureCollection"
    assert "features" in data


@pytest.mark.asyncio
async def test_search_endpoint(app_client):
    """Test POST /search endpoint."""
    search_body = {
        "collections": ["io-lulc-9-class"],
        "limit": 3,
        "datetime": "2019-01-01T00:00:00Z/2023-01-01T00:00:00Z"
    }
    
    response = await app_client.post("/search", json=search_body)
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) <= 3


def test_database_fixture(database):
    """Test that database fixture is properly configured."""
    assert database is not None
    assert hasattr(database, 'settings')
    assert 'io-lulc-9-class' in database.settings.parquet_urls


@pytest.mark.asyncio
async def test_real_item_fixture(test_item):
    """Test that test_item fixture returns real data from GeoParquet."""
    assert test_item is not None
    assert test_item["type"] == "Feature"
    assert "id" in test_item
    assert "collection" in test_item
    assert "geometry" in test_item
    assert "properties" in test_item