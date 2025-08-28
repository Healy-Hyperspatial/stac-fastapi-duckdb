"""DuckDB Filter Client for STAC API filtering."""

from typing import Any, Dict, List, Optional

from stac_fastapi.extensions.core.filter.client import AsyncBaseFiltersClient


class DuckDBFilterClient(AsyncBaseFiltersClient):
    """DuckDB-specific implementation of the filter client."""

    def __init__(self, database):
        """Initialize the DuckDB filter client."""
        self.database = database

    async def get_queryables(
        self, collection_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Get queryable properties for a collection.
        
        Args:
            collection_id: Optional collection ID to get queryables for
            
        Returns:
            Dict containing queryable properties schema
        """
        # Basic queryables for STAC items
        queryables = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$id": "https://stac-api.example.com/queryables",
            "type": "object",
            "title": "Queryables for STAC API",
            "description": "Queryable properties for STAC items",
            "properties": {
                "id": {
                    "description": "Item identifier",
                    "type": "string"
                },
                "collection": {
                    "description": "Collection identifier",
                    "type": "string"
                },
                "datetime": {
                    "description": "Item datetime",
                    "type": "string",
                    "format": "date-time"
                },
                "start_datetime": {
                    "description": "Item start datetime",
                    "type": "string",
                    "format": "date-time"
                },
                "end_datetime": {
                    "description": "Item end datetime", 
                    "type": "string",
                    "format": "date-time"
                }
            },
            "additionalProperties": True
        }
        
        return queryables

    async def get_filter_extension_name(self) -> str:
        """Get the name of the filter extension."""
        return "filter"

    async def get_supported_cql2_ops(self) -> List[str]:
        """Get list of supported CQL2 operators."""
        return [
            "eq", "neq", "lt", "lte", "gt", "gte",
            "and", "or", "not",
            "in", "between",
            "like", "ilike",
            "isNull"
        ]
