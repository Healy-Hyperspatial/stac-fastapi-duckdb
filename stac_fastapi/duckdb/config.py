"""API configuration."""
from typing import Set

from stac_fastapi.types.config import ApiSettings


class DuckDBSettings(ApiSettings):
    """MongoDB specific API settings."""

    forbidden_fields: Set[str] = {"type"}
    indexed_fields: Set[str] = {"datetime"}
