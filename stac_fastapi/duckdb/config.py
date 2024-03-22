"""API configuration."""
import os
from typing import Set

import duckdb
from fastapi import HTTPException
from stac_fastapi.types.config import ApiSettings


class DuckDBSettings(ApiSettings):
    """DuckDB specific API settings."""

    forbidden_fields: Set[str] = {"type"}
    indexed_fields: Set[str] = {"datetime"}

    @property
    def create_client(self):
        """Create duckdb client."""
        # Read Parquet file path from environment variable
        parquet_file_path = os.getenv(
            "PARQUET_FILE_PATH", "default_parquet_file_path.parquet"
        )

        # Check if the Parquet file exists at the given path
        if not os.path.exists(parquet_file_path):
            # Raise a 404 Not Found exception if the file doesn't exist
            raise HTTPException(
                status_code=404,
                detail=f"Parquet file not found at path: {parquet_file_path}",
            )

        try:
            # Connect to DuckDB
            conn = duckdb.connect(database=":memory:", read_only=False)

            # Load the Parquet file
            relation = conn.from_parquet(parquet_file_path)
            print(
                f"Successfully connected to DuckDB and loaded Parquet file from {parquet_file_path}"
            )
            return relation
        except Exception as e:
            # Raise a 500 Internal Server Error exception if loading the file fails
            raise HTTPException(
                status_code=500, detail=f"Failed to load Parquet file into DuckDB: {e}"
            )
