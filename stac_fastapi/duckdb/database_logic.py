"""Database logic."""
import json
import logging
import os
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pandas as pd
from fastapi import HTTPException, Request
from stac_fastapi.core import serializers
from stac_fastapi.extensions.core import SortExtension
from stac_fastapi.types.errors import NotFoundError  # ConflictError
from stac_fastapi.types.stac import Collection, Item

from stac_fastapi.duckdb.config import DuckDBSettings
from stac_fastapi.duckdb.utilities import create_stac_item

T = TypeVar("T")

logger = logging.getLogger(__name__)

NumType = Union[float, int]


class Geometry(Protocol):  # noqa
    type: str
    coordinates: Any


class DatabaseLogic:
    """Database logic for querying GeoParquet using per-request DuckDB connections."""

    def __init__(self, settings: Optional[DuckDBSettings] = None) -> None:
        """Initialize the database logic."""
        self.settings: DuckDBSettings = settings or DuckDBSettings()
        self.stac_file_path: str = os.getenv("STAC_FILE_PATH", "")
        self.item_serializer: Type[
            serializers.ItemSerializer
        ] = serializers.ItemSerializer
        self.collection_serializer: Type[
            serializers.CollectionSerializer
        ] = serializers.CollectionSerializer
        self.extensions: List[str] = []

    """CORE LOGIC"""

    # Connection creation and source resolution are provided by settings

    async def get_all_collections(
        self, token: Optional[str], limit: int, request: Request
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieve a list of all collections from the MongoDB database, supporting pagination.

        Args:
            token (Optional[str]): The pagination token, which is the ID of the last collection seen.
            limit (int): The maximum number of results to return.
            base_url (str): The base URL for constructing fully qualified links.

        Returns:
            Tuple[List[Dict[str, Any]], Optional[str]]: A tuple containing a list of collections
            and an optional next token for pagination.
        """
        collections = []

        if not os.path.exists(self.stac_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"STAC_FILE_PATH directory not found at path: {self.stac_file_path}",
            )

        # Iterate through each subdirectory under STAC_FILE_PATH to find collection.json files
        for collection_name in os.listdir(self.stac_file_path):
            collection_dir = os.path.join(self.stac_file_path, collection_name)
            if os.path.isdir(collection_dir):
                collection_json_path = os.path.join(collection_dir, "collection.json")
                if os.path.exists(collection_json_path):
                    try:
                        with open(collection_json_path, "r") as json_file:
                            collection = json.load(json_file)
                            serialized_collection = (
                                self.collection_serializer.db_to_stac(
                                    collection,
                                    request=request,
                                    extensions=self.extensions,
                                )
                            )
                            collections.append(serialized_collection)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {collection_json_path}")
                        continue
                else:
                    continue  # Skip directories without a collection.json file

        # Simulating pagination token
        next_token = None

        return collections, next_token

    async def find_collection(self, collection_id: str) -> dict:
        """
        Find and return a collection from the database.

        Args:
            self: The instance of the object calling this function.
            collection_id (str): The ID of the collection to be found.

        Returns:
            dict: The found collection, represented as a dictionary.

        Raises:
            NotFoundError: If the collection with the given `collection_id` is not found in the database.
        """
        collection_dir = os.path.join(self.stac_file_path, collection_id)
        collection_json_path = os.path.join(collection_dir, "collection.json")

        # Check if the collection.json file exists
        if not os.path.exists(collection_json_path):
            raise NotFoundError(f"Collection {collection_id} not found")

        try:
            with open(collection_json_path, "r") as json_file:
                collection = json.load(json_file)
                return collection
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Error decoding JSON from {collection_json_path}",
            )

    async def get_one_item(self, collection_id: str, item_id: str) -> dict:
        """Retrieve a single item from the database.

        Args:
            collection_id (str): The id of the Collection that the Item belongs to.
            item_id (str): The id of the Item.

        Returns:
            dict: A STAC item as a dictionary.

        Raises:
            HTTPException: If the specified Item does not exist in the Collection.
        """
        print(f"Get One Item: collection={collection_id}, item={item_id}")
        try:
            # Get the parquet URL for the collection
            sources = self.settings.resolve_sources([collection_id])
            if collection_id not in [src[0] for src in sources]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection {collection_id} not in configuration.",
                )

            _, url = sources[0]
            print(f"Using parquet URL: {url}")

            with self.settings.create_connection() as conn:
                # Use the same query pattern as in execute_search for consistency
                query = """
                    SELECT * FROM (
                        SELECT *, ? AS collection
                        FROM read_parquet(?)
                   )
                    WHERE id = ?
                    LIMIT 1
                """

                try:
                    df = conn.execute(query, [collection_id, url, item_id]).df()
                    if df.empty:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Item {item_id} in collection {collection_id} does not exist.",
                        )

                    # Create and return the STAC item
                    item = create_stac_item(
                        df=df, collection_id=collection_id, item_id=item_id
                    )
                    return item

                except Exception as db_error:
                    print(f"Database error in get_one_item: {str(db_error)}")
                    raise HTTPException(
                        status_code=500, detail=f"Error querying item: {str(db_error)}"
                    )

        except HTTPException:
            raise
        except Exception as e:
            print(f"Unexpected error in get_one_item: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @staticmethod
    def make_search():
        """Database logic to create a Search instance."""
        dict = {}
        return dict

    @staticmethod
    def apply_ids_filter(search: dict, item_ids: List[str]):
        """Database logic to search a list of STAC item ids."""
        search["item_ids"] = item_ids
        return search

    @staticmethod
    def apply_collections_filter(search: dict, collection_ids: List[str]):
        """Database logic to search a list of STAC collection ids."""
        search["collection_ids"] = collection_ids
        return search

    @staticmethod
    def apply_datetime_filter(
        search: dict, datetime: Optional[str] = None, interval: Optional[str] = None
    ) -> Tuple[dict, Optional[dict]]:
        """Apply a filter to search based on datetime, start_datetime, and end_datetime fields.

        This emulates the stac-fastapi-elasticsearch datetime filter logic for DuckDB:
        - For exact matches: include items with matching datetime OR items with null datetime
          where the time falls within their start_datetime/end_datetime range
        - For date ranges: include items with datetime in range OR items with null datetime
          that overlap the search range

        Args:
            search (dict): The search dictionary containing query parameters.
            datetime (Optional[str]): The datetime interval to filter by. Can be:
                - A single datetime string (e.g., "2023-01-01T12:00:00")
                - A datetime range string (e.g., "2023-01-01/2023-12-31")
                - None to skip filtering
            interval (Optional[str]): Legacy parameter name for datetime interval.

        Returns:
            tuple: (search dict, datetime_search dict) for compatibility with stac-fastapi-core 6.2.1
        """
        # Handle both new 'datetime' parameter and legacy 'interval' parameter
        datetime_value = datetime or interval
        if not datetime_value:
            return search, None

        # Parse the interval string into datetime_search format
        datetime_search = DatabaseLogic._parse_datetime_interval(datetime_value)
        if not datetime_search:
            return search, None

        # Initialize the filters list if it doesn't exist
        if "filters" not in search:
            search["filters"] = []

        # Build SQL conditions based on datetime filter type
        if "eq" in datetime_search:
            # Exact match case: emulate ES "should" logic with OR conditions
            exact_datetime = datetime_search["eq"]
            datetime_condition = f"""(
                (datetime IS NOT NULL AND datetime = '{exact_datetime}')
                OR 
                (datetime IS NULL 
                 AND start_datetime IS NOT NULL 
                 AND end_datetime IS NOT NULL
                 AND start_datetime <= '{exact_datetime}' 
                 AND end_datetime >= '{exact_datetime}')
            )"""
            search["filters"].append(datetime_condition)
        else:
            # Range case: handle gte/lte combinations
            gte_value = datetime_search.get("gte")
            lte_value = datetime_search.get("lte")

            if gte_value and lte_value:
                # Both start and end of range specified
                range_condition = f"""(
                    (datetime IS NOT NULL 
                     AND datetime >= '{gte_value}' 
                     AND datetime <= '{lte_value}')
                    OR 
                    (datetime IS NULL 
                     AND start_datetime IS NOT NULL 
                     AND end_datetime IS NOT NULL
                     AND start_datetime <= '{lte_value}' 
                     AND end_datetime >= '{gte_value}')
                )"""
                search["filters"].append(range_condition)
            elif gte_value:
                # Only start of range specified
                gte_condition = f"""(
                    (datetime IS NOT NULL AND datetime >= '{gte_value}')
                    OR 
                    (datetime IS NULL 
                     AND end_datetime IS NOT NULL
                     AND end_datetime >= '{gte_value}')
                )"""
                search["filters"].append(gte_condition)
            elif lte_value:
                # Only end of range specified
                lte_condition = f"""(
                    (datetime IS NOT NULL AND datetime <= '{lte_value}')
                    OR 
                    (datetime IS NULL 
                     AND start_datetime IS NOT NULL
                     AND start_datetime <= '{lte_value}')
                )"""
                search["filters"].append(lte_condition)

        # Return tuple for compatibility with stac-fastapi-core 6.2.1
        return search, datetime_search

    @staticmethod
    def _parse_datetime_interval(interval: str) -> dict:
        """Parse a datetime interval string into a search dictionary.

        Args:
            interval (str): The datetime interval. Can be:
                - A single datetime string (e.g., "2023-01-01T12:00:00")
                - A datetime range string (e.g., "2023-01-01/2023-12-31")
                - An open-ended range (e.g., "2023-01-01/.." or "../2023-12-31")

        Returns:
            dict: A dictionary with 'eq', 'gte', and/or 'lte' keys for filtering.
        """
        if not interval:
            return {}

        # Handle range intervals with "/" separator
        if "/" in interval:
            parts = interval.split("/", 1)
            start_date = parts[0].strip() if parts[0].strip() != ".." else None
            end_date = parts[1].strip() if parts[1].strip() != ".." else None

            datetime_search = {}
            if start_date:
                datetime_search["gte"] = start_date
            if end_date:
                datetime_search["lte"] = end_date
            return datetime_search
        else:
            # Single datetime - treat as exact match
            return {"eq": interval.strip()}

    @staticmethod
    def apply_bbox_filter(search: dict, bbox: List):
        """Filter search results based on bounding box.

        Args:
            search (dict): The search object to apply the filter to.
            bbox (List): The bounding box coordinates [west, south, east, north].

        Returns:
            dict: The search object with the bounding box filter applied.
        """
        # Ensure search is a dict
        if not isinstance(search, dict):
            logger.error(f"Expected search to be dict, got {type(search)}: {search}")
            return search if isinstance(search, dict) else {}

        if not bbox or len(bbox) != 4:
            return search

        # Ensure bbox is a list of floats
        # The bbox should already be converted to a list of floats by the core client
        # But we'll handle string input just in case
        if isinstance(bbox, str):
            try:
                logger.debug(f"Converting bbox string '{bbox}' to list of floats")
                bbox = [float(coord.strip()) for coord in bbox.split(",")]
            except (ValueError, AttributeError):
                logger.warning(
                    f"Invalid bbox format: {bbox}. Expected comma-separated list of 4 coordinates."
                )
                return search

        # Validate bbox format
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            logger.warning(f"Expected bbox with 4 coordinates, got {bbox}")
            return search

        try:
            west, south, east, north = map(float, bbox)
            logger.debug(
                f"Using bbox coordinates: west={west}, south={south}, east={east}, north={north}"
            )

            # Validate coordinates
            if not all(
                isinstance(coord, (int, float)) for coord in [west, south, east, north]
            ):
                logger.warning(f"Invalid bbox coordinates: {bbox}")
                return search

            # Create spatial filter using DuckDB's ST_Intersects with a bounding box polygon
            bbox_wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
            spatial_filter = f"ST_Intersects(geometry, ST_GeomFromText('{bbox_wkt}'))"
            logger.debug(f"Created spatial filter with WKT: {bbox_wkt}")

            # Add to filters list
            if "filters" not in search:
                search["filters"] = []
            search["filters"].append(spatial_filter)
            logger.info(f"Applied bbox filter with coordinates: {bbox}")

        except Exception as e:
            logger.error(f"Error applying bbox filter: {str(e)}")
            import traceback

            logger.debug(f"Bbox filter error traceback: {traceback.format_exc()}")

        return search

    @staticmethod
    def apply_intersects_filter(
        search: dict,
        intersects: Geometry,
    ):
        """Filter search results based on intersecting geometry.

        Args:
            search (dict): The search object to apply the filter to.
            intersects (Geometry): The intersecting geometry, represented as a GeoJSON-like object.

        Returns:
            dict: The search object with the intersecting geometry filter applied.
        """
        if not intersects:
            return search

        # Convert GeoJSON geometry to WKT for DuckDB
        import json

        geojson_str = json.dumps(
            {"type": intersects.type, "coordinates": intersects.coordinates}
        )
        spatial_filter = f"ST_Intersects(geometry, ST_GeomFromGeoJSON('{geojson_str}'))"

        # Add to filters list
        if "filters" not in search:
            search["filters"] = []
        search["filters"].append(spatial_filter)

        return search

    @staticmethod
    def apply_stacql_filter(search: dict, op: str, field: str, value: float):
        """Filter search results based on a comparison between a field and a value.

        Args:
            search (Search): The search object to apply the filter to.
            op (str): The comparison operator to use. Can be 'eq' (equal), 'gt' (greater than), 'gte' (greater than or equal),
                'lt' (less than), or 'lte' (less than or equal).
            field (str): The field to perform the comparison on.
            value (float): The value to compare the field against.

        Returns:
            search (Search): The search object with the specified filter applied.
        """
        pass
        # MongoDB comparison operators mapping
        # op_mapping = {
        #     "eq": "$eq",
        #     "gt": "$gt",
        #     "gte": "$gte",
        #     "lt": "$lt",
        #     "lte": "$lte",
        # }

        # # Replace double underscores with dots for nested field queries
        # field = field.replace("__", ".")

        # # Construct the MongoDB filter
        # if op in op_mapping:
        #     mongo_op = op_mapping[op]
        #     filter_condition = {field: {mongo_op: value}}
        # else:
        #     raise ValueError(f"Unsupported operation '{op}'")

        # # Add the constructed filter to the search adapter's filters
        # search.add_filter(filter_condition)
        # return search

    async def apply_cql2_filter(
        self, search: dict, _filter: Optional[Dict[str, Any]]
    ) -> dict:
        """
        Apply a CQL2 filter to a DuckDB search dictionary.

        This method transforms a dictionary representing a CQL2 filter into DuckDB SQL
        conditions and applies it to the provided search dictionary.

        Args:
            search (dict): The search dictionary containing query parameters.
            _filter (Optional[Dict[str, Any]]): The CQL2 filter in dictionary form.
                                                If None, the original search is returned.

        Returns:
            dict: Updated search dictionary with CQL2 filters applied.
        """
        if not _filter:
            return search

        try:
            # Convert CQL2 filter to SQL WHERE condition
            sql_condition = self._cql2_to_sql(_filter)
            if sql_condition:
                # Initialize filters if not present
                if "filters" not in search:
                    search["filters"] = []
                search["filters"].append(sql_condition)
        except Exception as e:
            logger.error(f"Error applying CQL2 filter: {str(e)}")
            # Return search unchanged if filter conversion fails
            return search

        return search

    def _cql2_to_sql(self, cql2_filter: Dict[str, Any]) -> Optional[str]:
        """
        Convert a CQL2 filter to SQL WHERE condition.

        Args:
            cql2_filter: CQL2 filter dictionary

        Returns:
            SQL WHERE condition string or None if conversion fails
        """
        try:
            return self._convert_cql2_expression(cql2_filter)
        except Exception as e:
            logger.warning(f"CQL2 to SQL conversion failed: {str(e)}")
            return None

    def _convert_cql2_expression(self, expr: Dict[str, Any]) -> str:
        """
        Recursively convert CQL2 expressions to SQL.

        Args:
            expr: CQL2 expression dictionary

        Returns:
            SQL condition string
        """
        if not isinstance(expr, dict):
            return str(expr)

        # Handle logical operators
        if "and" in expr:
            conditions = [self._convert_cql2_expression(sub) for sub in expr["and"]]
            return f"({' AND '.join(conditions)})"

        if "or" in expr:
            conditions = [self._convert_cql2_expression(sub) for sub in expr["or"]]
            return f"({' OR '.join(conditions)})"

        if "not" in expr:
            condition = self._convert_cql2_expression(expr["not"])
            return f"NOT ({condition})"

        # Handle comparison operators
        if "=" in expr:
            args = expr["="]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} = {right}"

        if "<>" in expr:
            args = expr["<>"]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} <> {right}"

        if "<" in expr:
            args = expr["<"]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} < {right}"

        if "<=" in expr:
            args = expr["<="]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} <= {right}"

        if ">" in expr:
            args = expr[">"]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} > {right}"

        if ">=" in expr:
            args = expr[">="]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} >= {right}"

        # Handle LIKE operator
        if "like" in expr:
            args = expr["like"]
            left, right = self._format_operands(args[0], args[1])
            return f"{left} LIKE {right}"

        # Handle IN operator
        if "in" in expr:
            args = expr["in"]
            left = self._format_field_name(args[0])
            values = [self._format_value(v) for v in args[1]]
            return f"{left} IN ({', '.join(values)})"

        # Handle BETWEEN operator
        if "between" in expr:
            args = expr["between"]
            field = self._format_field_name(args[0])
            lower = self._format_value(args[1])
            upper = self._format_value(args[2])
            return f"{field} BETWEEN {lower} AND {upper}"

        # Handle IS NULL
        if "isNull" in expr:
            field = self._format_field_name(expr["isNull"])
            return f"{field} IS NULL"

        # If we can't handle the expression, return a safe default
        logger.warning(f"Unsupported CQL2 expression: {expr}")
        return "1=1"  # Always true condition

    def _format_operands(self, left: Any, right: Any) -> tuple:
        """Format left and right operands for SQL comparison."""
        left_formatted = (
            self._format_field_name(left)
            if isinstance(left, dict) and "property" in left
            else self._format_value(left)
        )
        right_formatted = (
            self._format_field_name(right)
            if isinstance(right, dict) and "property" in right
            else self._format_value(right)
        )
        return left_formatted, right_formatted

    def _format_field_name(self, field: Any) -> str:
        """Format a field name for SQL."""
        if isinstance(field, dict) and "property" in field:
            return field["property"]
        return str(field)

    def _format_value(self, value: Any) -> str:
        """Format a value for SQL."""
        if isinstance(value, str):
            # Escape single quotes and wrap in quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif value is None:
            return "NULL"
        else:
            return f"'{str(value)}'"

    @staticmethod
    def populate_sort(sortby: List[SortExtension]) -> List[Tuple[str, int]]:
        """
        Transform a list of sort criteria into the format expected by MongoDB.

        Args:
            sortby (List[SortExtension]): A list of SortExtension objects with 'field'
                                        and 'direction' attributes.

        Returns:
            List[Tuple[str, int]]: A list of tuples where each tuple is (fieldname, direction),
                                with direction being 1 for 'asc' and -1 for 'desc'.
                                Returns an empty list if no sort criteria are provided.
        """
        return sortby
        # if not sortby:
        #     return []

        # mongo_sort = []
        # for sort_extension in sortby:
        #     field = sort_extension.field
        #     # Convert the direction enum to a string, then to MongoDB's expected format
        #     direction = 1 if sort_extension.direction.value == "asc" else -1
        #     mongo_sort.append((field, direction))

        # return mongo_sort

    async def get_total_count(
        self,
        search: dict,
        collection_ids: Optional[List[str]] = None,
    ) -> Optional[int]:
        """Get the total count of items matching the search criteria.

        Args:
            search (dict): The search dictionary containing filters
            collection_ids (Optional[List[str]]): Collection IDs to search, or None for all

        Returns:
            Optional[int]: Total count of matching items, or None if count failed
        """
        try:
            # If no collection IDs provided, search all available collections
            if not collection_ids:
                all_collection_ids = list(self.settings.parquet_urls.keys())
                if not all_collection_ids:
                    return 0
                collection_ids = all_collection_ids

            # Resolve sources for the collections
            sources = self.settings.resolve_sources(collection_ids)

            # Get filters from search
            item_ids: Optional[List[str]] = (
                search.get("item_ids") if isinstance(search, dict) else None
            )

            # Build count query for each collection
            count_subqueries: List[str] = []
            count_params: List[Any] = []

            for cid, url in sources:
                count_sq = "SELECT COUNT(*) as count FROM read_parquet(?)"
                count_params.append(url)
                count_wheres: List[str] = []

                # Add item_ids filter if specified
                if item_ids:
                    placeholders = ", ".join(["?"] * len(item_ids))
                    count_wheres.append(f"id IN ({placeholders})")
                    count_params.extend(item_ids)

                # Add datetime and other filters if specified
                if "filters" in search and search["filters"]:
                    count_wheres.extend(search["filters"])

                if count_wheres:
                    count_sq += " WHERE " + " AND ".join(count_wheres)
                count_subqueries.append(count_sq)

            # Execute count query
            if not count_subqueries:
                return 0

            count_union_sql = " UNION ALL ".join(count_subqueries)
            final_count_sql = (
                f"SELECT SUM(count) as total_count FROM ({count_union_sql})"
            )

            with self.settings.create_connection() as conn:
                count_result = conn.execute(final_count_sql, count_params).fetchone()
                return (
                    int(count_result[0])
                    if count_result and count_result[0] is not None
                    else 0
                )

        except Exception as e:
            logger.warning(f"Failed to calculate total count: {str(e)}")
            return None

    async def execute_search(
        self,
        search: dict,
        limit: int,
        token: Optional[str],
        sort: Optional[Dict[str, Dict[str, str]]],
        collection_ids: Optional[List[str]],
        ignore_unavailable: bool = True,
        datetime_search: Optional[dict] = None,
    ) -> Tuple[Iterable[Dict[str, Any]], Optional[int], Optional[str]]:
        """Execute a search query with limit and other optional parameters.

        Args:
            search (Search): The search query to be executed.
            limit (int): The maximum number of results to be returned.
            token (Optional[str]): The token used to return the next set of results.
            sort (Optional[Dict[str, Dict[str, str]]]): Specifies how the results should be sorted.
            collection_ids (Optional[List[str]]): The collection ids to search.
            ignore_unavailable (bool, optional): Whether to ignore unavailable collections. Defaults to True.

        Returns:
            Tuple[Iterable[Dict[str, Any]], Optional[int], Optional[str]]: A tuple containing:
                - An iterable of search results, where each result is a dictionary with keys and values representing the
                fields and values of each document.
                - The total number of results (if the count could be computed), or None if the count could not be
                computed.
                - The token to be used to retrieve the next set of results, or None if there are no more results.

        Raises:
            NotFoundError: If the collections specified in `collection_ids` do not exist.
        """
        print("limit", limit)
        print("token", token)
        print("sort", sort)
        print("collection_ids", collection_ids)
        print("ignore_unavailable", ignore_unavailable)

        # If no collection IDs provided, search all available collections
        if not collection_ids:
            # Get all available collection IDs from the settings
            all_collection_ids = list(self.settings.parquet_urls.keys())
            if not all_collection_ids:
                raise HTTPException(
                    status_code=404, detail="No collections available for search."
                )
            collection_ids = all_collection_ids
            print(
                f"No collections specified, searching all available: {collection_ids}"
            )

        # Resolve sources for the collections
        sources = self.settings.resolve_sources(collection_ids)

        # Basic filters
        item_ids: Optional[List[str]] = (
            search.get("item_ids") if isinstance(search, dict) else None
        )

        # Build dynamic SQL using UNION ALL over sources
        subqueries: List[str] = []
        params: List[Any] = []
        for cid, url in sources:
            sq = "SELECT *, ? AS collection FROM read_parquet(?)"
            params.extend([cid, url])
            wheres: List[str] = []

            # Add item_ids filter if specified
            if item_ids:
                placeholders = ", ".join(["?"] * len(item_ids))
                wheres.append(f"id IN ({placeholders})")
                params.extend(item_ids)

            # Add datetime filters if specified
            if "filters" in search and search["filters"]:
                wheres.extend(search["filters"])

            if wheres:
                sq += " WHERE " + " AND ".join(wheres)
            subqueries.append(sq)

        union_sql = " UNION ALL ".join(subqueries)
        base_sql = f"SELECT * FROM ({union_sql})"

        # Sorting
        if sort:
            # Handle different sort formats
            if "field" in sort and "direction" in sort:
                # Format: {"field": "id", "direction": "asc"}
                field = sort["field"]
                direction = sort["direction"]
                sort_clause = f"{field} {direction}"
            else:
                # Format: {"field1": {"order": "asc"}, "field2": {"order": "desc"}}
                sort_clause = ", ".join(
                    f"{field} {direction['order']}" for field, direction in sort.items()
                )
            base_sql += f" ORDER BY {sort_clause}"

        # Pagination: mimic ES search_after pattern with offset-based approach
        offset_val = 0
        if token:
            try:
                # For now, use simple integer offset (can be enhanced to base64 later)
                offset_val = int(token)
            except (ValueError, TypeError):
                offset_val = 0

        # Request one more item than limit to check if there are more results
        size_limit = limit + 1 if limit else None
        if size_limit:
            base_sql += " LIMIT ? OFFSET ?"
            params.extend([size_limit, offset_val])

        # Execute the main search query
        try:
            with self.settings.create_connection() as conn:
                actual_results = conn.execute(base_sql, params).fetchdf()
        except Exception as e:
            if "not found" in str(e).lower():
                from stac_fastapi.types.errors import NotFoundError

                raise NotFoundError(f"Collections '{collection_ids}' do not exist")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        # Check if there are more results
        has_more = len(actual_results) > limit if limit else False
        if has_more:
            # Remove the extra row we fetched
            actual_results = actual_results.iloc[:-1]

        next_token = None
        if has_more and limit:
            next_token = str(offset_val + limit)

        # Calculate total count for numMatched using the dedicated function
        total = await self.get_total_count(search, collection_ids)

        # Convert each row to a proper STAC item using create_stac_item
        items = []
        failed_items = 0
        for idx, row in actual_results.iterrows():
            item_id = row.get("id", "")
            collection_id = row.get("collection", "") or row.get("collection_1", "")
            try:
                # Create a DataFrame with a single row for this item
                row_df = pd.DataFrame([row])
                item = create_stac_item(
                    df=row_df, item_id=item_id, collection_id=collection_id
                )
                if not item or not isinstance(item, dict):
                    logger.warning(
                        f"create_stac_item returned invalid item for {item_id}: {type(item)}"
                    )
                    failed_items += 1
                    continue
                items.append(item)
            except Exception as e:
                logger.error(
                    f"Failed to create STAC item for {item_id} (row {idx}): {str(e)}"
                )
                try:
                    # Safely log row data types
                    if hasattr(row, "dtypes"):
                        dtypes_info = {col: str(row.dtypes[col]) for col in row.index}
                        logger.error(f"Row data types: {dtypes_info}")
                    else:
                        logger.error("No dtypes available")

                    # Log sample data safely
                    sample_data = {}
                    for col in row.index if hasattr(row, "index") else row.keys():
                        try:
                            value = row[col]
                            sample_data[
                                col
                            ] = f"{type(value).__name__}: {str(value)[:50]}..."
                        except Exception:
                            sample_data[col] = "Error accessing value"
                    logger.error(f"Row sample data: {sample_data}")

                    # Log specific problematic fields
                    if "geometry" in row and row.get("geometry") is not None:
                        geom_val = row.get("geometry")
                        logger.error(
                            f"Geometry type: {type(geom_val)}, value: {str(geom_val)[:200]}..."
                        )
                except Exception as log_error:
                    logger.error(f"Error in logging row data: {str(log_error)}")
                failed_items += 1

        # Log summary of item conversion
        if failed_items > 0:
            logger.warning(
                f"Failed to convert {failed_items} out of {len(actual_results)} items to STAC format"
            )
            logger.info(
                f"Successfully returned {len(items)} items out of {len(actual_results)} rows"
            )

        return items, total, next_token

    """ TRANSACTION LOGIC """

    async def check_collection_exists(self, collection_id: str):
        """
        Check if a specific STAC collection exists within the MongoDB database.

        This method queries the MongoDB collection specified by COLLECTIONS_INDEX to determine
        if a document with the specified collection_id exists.

        Args:
            collection_id (str): The ID of the STAC collection to check for existence.

        Raises:
            NotFoundError: If the STAC collection specified by `collection_id` does not exist
                        within the MongoDB collection defined by COLLECTIONS_INDEX.
        """
        pass
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]

        # # Query the collections collection to see if a document with the specified collection_id exists
        # collection_exists = await collections_collection.find_one({"id": collection_id})
        # if not collection_exists:
        #     raise NotFoundError(f"Collection {collection_id} does not exist")

    async def create_item(self, item: Item, refresh: bool = False):
        """
        Asynchronously inserts a STAC item into MongoDB, ensuring the item does not already exist.

        Args:
            item (Item): The STAC item to be created.
            refresh (bool, optional): Not used for MongoDB, kept for compatibility with Elasticsearch interface.

        Raises:
            ConflictError: If the item with the same ID already exists within the collection.
            NotFoundError: If the specified collection does not exist in MongoDB.
        """
        pass

    async def prep_create_item(
        self, item: Item, base_url: str, exist_ok: bool = False
    ) -> Item:
        """
        Preps an item for insertion into the MongoDB database.

        Args:
            item (Item): The item to be prepped for insertion.
            base_url (str): The base URL used to create the item's self URL.
            exist_ok (bool): Indicates whether the item can exist already.

        Returns:
            Item: The prepped item.

        Raises:
            ConflictError: If the item already exists in the database and exist_ok is False.
            NotFoundError: If the collection specified by the item does not exist.
        """
        pass

    def sync_prep_create_item(
        self, item: Item, base_url: str, exist_ok: bool = False
    ) -> Item:
        """
        Preps an item for insertion into the MongoDB database in a synchronous manner.

        Args:
            item (Item): The item to be prepped for insertion.
            base_url (str): The base URL used to create the item's self URL.
            exist_ok (bool): Indicates whether the item can exist already.

        Returns:
            Item: The prepped item.

        Raises:
            ConflictError: If the item already exists in the database and exist_ok is False.
            NotFoundError: If the collection specified by the item does not exist.
        """
        pass

    async def delete_item(
        self, item_id: str, collection_id: str, refresh: bool = False
    ):
        """
        Delete a single item from the database.

        Args:
            item_id (str): The id of the Item to be deleted.
            collection_id (str): The id of the Collection that the Item belongs to.
            refresh (bool, optional): Whether to refresh the index after the deletion. Default is False.

        Raises:
            NotFoundError: If the Item does not exist in the database.
        """
        pass

    async def create_collection(self, collection: Collection, refresh: bool = False):
        """Create a single collection document in the database.

        Args:
            collection (Collection): The Collection object to be created.
            refresh (bool, optional): Whether to refresh the index after the creation. Default is False.

        Raises:
            ConflictError: If a Collection with the same id already exists in the database.
        """
        pass

    async def update_collection(
        self, collection_id: str, collection: Collection, refresh: bool = False
    ):
        """
        Update a collection in the MongoDB database.

        Args:
            collection_id (str): The ID of the collection to be updated.
            collection (Collection): The new collection data to update.
            refresh (bool): Not applicable for MongoDB, kept for compatibility.

        Raises:
            NotFoundError: If the collection with the specified ID does not exist.
            ConflictError: If attempting to change the collection ID to one that already exists.

        Note:
            This function handles both updating a collection's metadata and changing its ID.
            It does not directly modify the `_id` field, which is immutable in MongoDB.
            When changing a collection's ID, it creates a new document with the new ID and deletes the old document.
        """
        pass

    async def delete_collection(self, collection_id: str):
        """
        Delete a collection from the MongoDB database and all items associated with it.

        This function first attempts to delete the specified collection from the database.
        If the collection exists and is successfully deleted, it then proceeds to delete
        all items that are associated with this collection. If the collection does not exist,
        a NotFoundError is raised to indicate the collection cannot be found in the database.

        Args:
            collection_id (str): The ID of the collection to be deleted.

        Raises:
            NotFoundError: If the collection with the specified ID does not exist in the database.

        This ensures that when a collection is deleted, all of its items are also cleaned up from the database,
        maintaining data integrity and avoiding orphaned items without a parent collection.
        """
        pass

    async def delete_items(self) -> None:
        """
        Danger. this is only for tests.

        Deletes all items from the 'items' collection in MongoDB.
        """
        pass

    async def delete_collections(self) -> None:
        """
        Danger. this is only for tests.

        Deletes all collections from the 'collections' collection in MongoDB.
        """
        pass
