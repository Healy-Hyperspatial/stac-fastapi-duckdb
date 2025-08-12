"""DuckDB runtime configuration and data source mapping."""
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb
from stac_fastapi.core.base_settings import ApiBaseSettings

# from stac_fastapi.core.utilities import get_bool_env
from stac_fastapi.types.config import ApiSettings


class DuckDBSettings(ApiSettings, ApiBaseSettings):
    """DuckDB API settings and configuration."""

    # Core API settings
    forbidden_fields: Set[str] = {"id", "type", "collection"}
    indexed_fields: Set[str] = {
        "datetime",
        "start_datetime",
        "end_datetime",
        "geometry",
    }
    # enable_response_models: bool = False

    # DuckDB-specific settings
    http_cache_path: str = os.getenv("HTTP_CACHE_PATH", "/tmp/duckdb_http_cache")
    stac_file_path: str = os.getenv("STAC_FILE_PATH", "/app/stac_collections")
    parquet_urls_json: str = os.getenv("PARQUET_URLS_JSON", "{}")
    _parquet_urls: Dict[str, str] = {}

    def __init__(self, **data: Any) -> None:
        """Initialize the settings."""
        super().__init__(**data)
        self._parquet_urls = json.loads(self.parquet_urls_json)

        # Ensure cache directory exists
        os.makedirs(self.http_cache_path, exist_ok=True)

        # Validate STAC file path if set
        if self.stac_file_path and not os.path.isdir(self.stac_file_path):
            raise ValueError(f"STAC file path does not exist: {self.stac_file_path}")

    @property
    def parquet_urls(self) -> Dict[str, str]:
        """Get the configured Parquet URLs."""
        return self._parquet_urls

    @parquet_urls.setter
    def parquet_urls(self, value: Union[Dict[str, str], str]) -> None:
        """Set Parquet URLs from either a dict or JSON string."""
        if isinstance(value, str):
            self._parquet_urls = json.loads(value)
        else:
            self._parquet_urls = value

    @property
    def database_refresh(self) -> Union[bool, str]:
        """Get the refresh setting for database operations."""
        return None

    def create_client(self):
        """Create a synchronous DuckDB client."""
        # Import here to avoid circular imports
        from stac_fastapi.duckdb.database_logic import DuckDBClient

        return DuckDBClient(settings=self)

    def get_collection_parquet_url(self, collection_id: str) -> str:
        """Get the Parquet URL for a collection."""
        if collection_id not in self._parquet_urls:
            raise ValueError(
                f"No Parquet URL configured for collection: {collection_id}"
            )
        return self._parquet_urls[collection_id]

    def get_collection_parquet_sources(
        self, collection_ids: Optional[list[str]] = None
    ) -> list[tuple[str, str]]:
        """Get a list of (collection_id, parquet_url) tuples."""
        if not collection_ids:
            collection_ids = list(self._parquet_urls.keys())

        sources = []
        for cid in collection_ids:
            url = self._parquet_urls.get(cid)
            if not url:
                raise ValueError(f"No Parquet configured for collection '{cid}'")
            sources.append((cid, url))
        return sources

    def resolve_sources(
        self, collection_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """Resolve collection ids to (collection_id, parquet_url) pairs.

        Thin wrapper used by DatabaseLogic.
        """
        pairs = self.get_collection_parquet_sources(collection_ids)
        return [(cid, url) for cid, url in pairs]

    @contextmanager
    def create_connection(self):
        """Create a per-request DuckDB connection with httpfs and basic caching configured."""
        conn = duckdb.connect(database=":memory:")
        try:
            # Enable remote I/O via httpfs where available
            try:
                conn.execute("INSTALL httpfs;")
            except Exception:
                pass
            try:
                conn.execute("LOAD httpfs;")
            except Exception:
                pass
            # Best-effort caching knobs
            try:
                conn.execute("SET enable_http_metadata_cache=true")
                conn.execute("SET enable_object_cache=true")
                conn.execute(f"SET http_metadata_cache='{self.http_cache_path}'")
            except Exception:
                pass
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass
