"""Duckdb config file."""
import os
from typing import Any, Set

import duckdb
from fastapi import HTTPException
from pydantic import validator
from stac_fastapi.types.config import ApiSettings

_forbidden_fields: Set[str] = {"type"}


class DuckDBSettings(ApiSettings):
    """Class."""

    forbidden_fields: Set[str] = _forbidden_fields
    indexed_fields: Set[str] = {"datetime"}

    conn: Any = None
    relation: Any = None

    _instance = None  # Private class variable to hold the singleton instance

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    @validator("conn", pre=True, always=True)
    def setup_connection(cls, v):
        """Set DuckDB connection, load spatial extension."""
        try:
            # Attempt a simple operation to check if the connection is alive
            if cls._instance is None or (
                v is None or not v.execute("SELECT 1").fetchall()
            ):
                # Connect to DuckDB, install and load the spatial extension
                conn = duckdb.connect(database=":memory:", read_only=False)
                conn.execute("INSTALL spatial;")
                conn.execute("LOAD spatial;")
                return conn
        except (Exception, duckdb.Error) as e:
            # If there is an error, reconnect and load spatial again
            print(e)
            conn = duckdb.connect(database=":memory:", read_only=False)
            conn.execute("INSTALL spatial;")
            conn.execute("LOAD spatial;")
            return conn
        return v

    @validator("relation", pre=True, always=True)
    def load_parquet_file(cls, v, values):
        """
        Load a Parquet file into a DuckDB table, creating or replacing the table.

        This method checks if the singleton instance or the parameter 'v' is None, and
        then tries to load a Parquet file into a DuckDB table specified by the environment
        variable 'PARQUET_FILE_PATH'. It handles creation or replacement of the table
        'items' with the data from the Parquet file.

        Parameters:
            v: The initial value or object; if this is None, the method proceeds with loading.
            values: Dictionary expected to have a 'conn' key with a DuckDB connection object.

        Returns:
            If successful, returns a DuckDB table object representing the loaded data.
            If 'v' is not None and no loading is needed, returns 'v'.

        Raises:
            HTTPException: If the Parquet file path is not found or if loading the Parquet
            file into DuckDB fails. Uses status code 404 for file not found and 500 for
            any other errors during file loading.
        """
        if cls._instance is None or v is None:
            parquet_file_path = os.getenv("PARQUET_FILE_PATH", "")
            if not os.path.exists(parquet_file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Parquet file not found at path: {parquet_file_path}",
                )
            conn = values.get("conn")
            if conn is not None:
                try:
                    table_name = "items"  # Set your desired table name here
                    # Drop existing table and create a new one from the Parquet file
                    conn.execute(f"DROP TABLE IF EXISTS {table_name};")
                    conn.execute(
                        f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_file_path}');"
                    )
                    return conn.table(table_name)  # Return the new table as a relation
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load Parquet file into DuckDB: {e}",
                    )
        return v

    @classmethod
    def get_instance(cls):
        """Get instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
