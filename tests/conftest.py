import asyncio
import os
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from stac_fastapi.api.app import StacApi
from stac_fastapi.core.core import CoreClient

from stac_fastapi.duckdb.app import app_config
from stac_fastapi.duckdb.config import DuckDBSettings
from stac_fastapi.duckdb.database_logic import DatabaseLogic

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Context:
    def __init__(self, item, collection):
        self.item = item
        self.collection = collection


class MockRequest:
    base_url = "http://test-server"
    url = "http://test-server/test"
    headers: Dict[str, str] = {}
    query_params: Dict[str, str] = {}

    def __init__(
        self,
        method: str = "GET",
        url: str = "XXXX",
        app: Optional[Any] = None,
        query_params: Dict[str, Any] = {"limit": "10"},
        headers: Dict[str, Any] = {"content-type": "application/json"},
    ):
        self.method = method
        self.url = url
        self.app = app
        self.query_params = query_params
        self.headers = headers


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def test_collection() -> Dict:
    """Real STAC collection for testing - io-lulc-9-class."""
    return {
        "type": "Collection",
        "id": "io-lulc-9-class",
        "stac_version": "1.0.0",
        "description": "Impact Observatory Land Use Land Cover 9-class",
        "license": "proprietary",
        "extent": {
            "spatial": {"bbox": [[-180, -90, 180, 90]]},
            "temporal": {
                "interval": [["2017-01-01T00:00:00Z", "2023-12-31T23:59:59Z"]]
            },
        },
        "links": [],
    }


@pytest.fixture
def settings():
    """DuckDB settings for testing with real GeoParquet file."""
    return DuckDBSettings(
        parquet_urls_json='{"io-lulc-9-class": "/app/stac_collections/io-lulc-9-class/io-lulc-9-class.parquet"}',
        stac_file_path="/app/stac_collections",
    )


@pytest.fixture
def database(settings):
    """Database logic instance for testing."""
    return DatabaseLogic(settings=settings)


@pytest.fixture
def core_client(database):
    """Core client for testing."""
    return CoreClient(database=database, session=None)


@pytest_asyncio.fixture
async def test_item(database):
    """Get a real STAC item from the GeoParquet file."""
    # Query the first item from the GeoParquet file
    items, total, next_token = await database.execute_search(
        collection_ids=["io-lulc-9-class"],
        search={},
        limit=1,
        token=None,
        sort={"field": "id", "direction": "asc"},
    )
    if items:
        return items[0]
    else:
        # Fallback mock item if GeoParquet file is not available
        return {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test-item-fallback",
            "collection": "io-lulc-9-class",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[-66, -16], [-60, -16], [-60, -8], [-66, -8], [-66, -16]]
                ],
            },
            "bbox": [-66, -16, -60, -8],
            "properties": {
                "start_datetime": "2022-01-01T00:00:00Z",
                "end_datetime": "2022-12-31T23:59:59Z",
            },
            "assets": {},
            "links": [],
        }


@pytest_asyncio.fixture(scope="session")
async def app():
    """FastAPI app instance for testing."""
    return StacApi(**app_config).app


@pytest_asyncio.fixture(scope="session")
async def app_client(app):
    """HTTP client for testing the app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test-server"
    ) as c:
        yield c


@pytest_asyncio.fixture()
async def ctx(test_collection, test_item):
    """Test context with sample collection and item."""
    yield Context(item=test_item, collection=test_collection)
