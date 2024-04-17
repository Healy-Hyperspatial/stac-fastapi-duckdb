import duckdb
import os
from pydantic import BaseModel, validator
from fastapi import HTTPException
from typing import Set, Any
from stac_fastapi.types.config import ApiSettings

_forbidden_fields: Set[str] = {"type"}

class DuckDBSettings(ApiSettings):
    # Fields which are defined by STAC but not included in the database model
    forbidden_fields: Set[str] = _forbidden_fields
    indexed_fields: Set[str] = {"datetime"}

    conn: Any = None
    relation: Any = None

    _instance = None  # Private class variable to hold the singleton instance

    class Config:
        arbitrary_types_allowed = True

    @validator('conn', pre=True, always=True)
    def setup_connection(cls, v):
        try:
            # Attempt a simple operation to check if the connection is alive
            if cls._instance is None or (v is None or not v.execute("SELECT 1").fetchall()):
                # Connect to DuckDB, install and load the spatial extension
                conn = duckdb.connect(database=":memory:", read_only=False)
                conn.execute("INSTALL spatial;")
                conn.execute("LOAD spatial;")
                return conn
        except (Exception, duckdb.Error) as e:
            # If there is an error, reconnect and load spatial again
            conn = duckdb.connect(database=":memory:", read_only=False)
            conn.execute("INSTALL spatial;")
            conn.execute("LOAD spatial;")
            return conn
        return v

    @validator('relation', pre=True, always=True)
    def load_parquet_file(cls, v, values):
        if cls._instance is None or v is None:
            parquet_file_path = os.getenv("PARQUET_FILE_PATH", "")
            if not os.path.exists(parquet_file_path):
                raise HTTPException(status_code=404, detail=f"Parquet file not found at path: {parquet_file_path}")
            conn = values.get('conn')
            if conn is not None:
                try:
                    table_name = "items"  # Set your desired table name here
                    # Drop existing table and create a new one from the Parquet file
                    conn.execute(f"DROP TABLE IF EXISTS {table_name};")
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_file_path}');")
                    return conn.table(table_name)  # Return the new table as a relation
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load Parquet file into DuckDB: {e}")
        return v

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def close(self):
        if self.conn:
            self.conn.close()
