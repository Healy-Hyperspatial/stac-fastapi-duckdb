"""Database logic."""
import base64
from datetime import datetime, date, timedelta
import json
import logging
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Type, Union

import attr
from fastapi import HTTPException
from stac_fastapi.core import serializers

# from stac_fastapi.core.extensions import filter
# from stac_fastapi.core.utilities import bbox2polygon
from shapely.wkb import loads
from shapely.geometry import mapping

from stac_fastapi.extensions.core import SortExtension
from stac_fastapi.types.errors import NotFoundError  # ConflictError
from stac_fastapi.types.stac import Collection, Item

from stac_fastapi.duckdb.config import DuckDBSettings

logger = logging.getLogger(__name__)

NumType = Union[float, int]


def mk_item_id(item_id: str, collection_id: str):
    """Create the document id for an Item in Elasticsearch.

    Args:
        item_id (str): The id of the Item.
        collection_id (str): The id of the Collection that the Item belongs to.

    Returns:
        str: The document id for the Item, combining the Item id and the Collection id, separated by a `|` character.
    """
    return f"{item_id}|{collection_id}"


class Geometry(Protocol):  # noqa
    type: str
    coordinates: Any


class DuckDBSearchAdapter:
    """
    Adapter class to manage search filters and sorting for MongoDB queries.

    Attributes:
        filters (list): A list of filter conditions to be applied to the MongoDB query.
        sort (list): A list of tuples specifying field names and their corresponding sort directions
                     for DuckDB sorting.

    Methods:
        add_filter(filter_condition): Adds a new filter condition to the filters list.
        set_sort(sort_conditions): Sets the sorting criteria based on a dictionary of field names
                                   and sort directions.
    """

    def __init__(self):
        """
        Initialize the DuckDBSearchAdapter with default sorting criteria.

        The default sort order is by 'properties.datetime' in descending order, followed by 'id' in descending order,
        and finally by 'collection' in descending order. This matches typical STAC item queries where the most recent items
        are retrieved first.
        """
        self.filters = []
        # self.sort = [("properties.datetime", -1), ("id", -1), ("collection", -1)]

    def add_filter(self, filter_condition):
        """
        Add a filter condition to the query.

        This method appends a new filter condition to the list of existing filters. Each filter condition
        should be a dictionary representing a MongoDB query condition.

        Args:
            filter_condition (dict): A dictionary representing a MongoDB filter condition.
        """
        self.filters.append(filter_condition)


@attr.s(auto_attribs=True)
class DatabaseLogic:
    """Database logic managing DuckDB connections."""
    settings = DuckDBSettings.get_instance()  # Access the singleton instance
    conn = settings.conn 
    stac_file_path: str = os.getenv("STAC_FILE_PATH", "")
    item_serializer: Type[serializers.ItemSerializer] = serializers.ItemSerializer
    collection_serializer: Type[serializers.CollectionSerializer] = serializers.CollectionSerializer

    """CORE LOGIC"""

    async def get_all_collections(
        self, token: Optional[str], limit: int, base_url: str
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
                                    collection, base_url
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
        
    def decode_geometry(self, int_list):
        # Convert the integer list to bytes
        geom_bytes = bytes(int_list)
        
        # Use Shapely to decode the WKB format
        geometry_object = loads(geom_bytes)
        
        # Convert to GeoJSON format
        return mapping(geometry_object)

    def convert_type(self, value):
        """Convert unsupported numpy types to native Python types for JSON serialization."""
        try:
            if isinstance(value, np.ndarray):
                return value.tolist()  # Convert numpy arrays to list
            if pd.isna(value):  # Check if the value is NaN or NaT
                return None 
            if isinstance(value, pd.Timestamp):
                return value.isoformat()  # Convert Timestamp to ISO format string
            elif isinstance(value, np.generic):
                if np.issubdtype(value.dtype, np.floating):
                    return float(value.item())
                elif np.issubdtype(value.dtype, np.integer):
                    return int(value.item())
                elif np.issubdtype(value.dtype, np.bool_):
                    return bool(value.item())
                elif np.issubdtype(value.dtype, np.datetime64):
                    return pd.to_datetime(value).isoformat()  # Ensure datetime64 types are also converted
                elif np.issubdtype(value.dtype, np.complex):
                    return str(value.item())  # Convert complex numbers to string
                else:
                    return value.item()  # Last resort, convert directly
            elif isinstance(value, (datetime, date)):
                return value.isoformat()  # Additional handling for native Python datetime types
            return value
        except Exception as e:
            print(f"Failed conversion for value {value} of type {type(value)}: {e}")
            return None  # Or choose to return a default value or placeholder

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
        try:
            query = "SELECT * FROM items WHERE id = ? LIMIT 1"

            df = self.conn.execute(query, [item_id]).df()
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Item {item_id} in collection {collection_id} does not exist.")

            print(df)
            print("HI")
            # Decode the WKB geometry to a shapely object
            geom_int_list = df.at[0, 'geometry']
            geojson_geometry = self.decode_geometry(geom_int_list)

            print(geojson_geometry)
            print("HI")
            # Initialize the STAC item dictionary
            item = {
                "type": "Feature",
                "stac_version": "1.0.0",
                "stac_extensions":  self.convert_type(df.at[0, "stac_extensions"]),
                "id": item_id,
                "bbox": df.at[0, "bbox"],
                "collection": collection_id,
                "properties": {},
                "geometry": geojson_geometry,
                "assets": {},
                "links": []
            }

            print(item)

            # Dynamically populate properties with improved error handling
            for column in df.columns:
                print(column)
                if column not in ['id', 'geometry', 'assets', 'links', 'type', 'bbox', 'stac_version', 'stac_extensions']:  # Exclude known non-property fields
                # if column in ['start_datetime', 'end_datetime', 'datetime', 'proj:epsg', 'proj:bbox']:
                    print(df.at[0, column])
                    converted_value = self.convert_type(df.at[0, column])
                    item['properties'][column] = converted_value if converted_value is not None else None 

            return item
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def make_search():
        """Database logic to create a Search instance."""
        return DuckDBSearchAdapter()

    @staticmethod
    def apply_ids_filter(search: DuckDBSearchAdapter, item_ids: List[str]):
        """Database logic to search a list of STAC item ids."""
        pass
        # search.add_filter({"id": {"$in": item_ids}})
        # return search

    @staticmethod
    def apply_collections_filter(search: DuckDBSearchAdapter, collection_ids: List[str]):
        """Database logic to search a list of STAC collection ids."""
        pass
        # search.add_filter({"collection": {"$in": collection_ids}})
        # return search

    @staticmethod
    def apply_datetime_filter(search: DuckDBSearchAdapter, datetime_search):
        """Apply a filter to search based on datetime field.

        Args:
            search (Search): The search object to filter.
            datetime_search (dict): The datetime filter criteria.

        Returns:
            Search: The filtered search object.
        """
        pass
        # if "eq" in datetime_search:
        #     search.add_filter({"properties.datetime": datetime_search["eq"]})
        # else:
        #     if "gte" in datetime_search:
        #         search.add_filter(
        #             {"properties.datetime": {"$gte": datetime_search["gte"]}}
        #         )
        #     if "lte" in datetime_search:
        #         search.add_filter(
        #             {"properties.datetime": {"$lte": datetime_search["lte"]}}
        #         )
        # return search

    @staticmethod
    def apply_bbox_filter(search: DuckDBSearchAdapter, bbox: List):
        """Filter search results based on bounding box.

        Args:
            search (Search): The search object to apply the filter to.
            bbox (List): The bounding box coordinates, represented as a list of four values [minx, miny, maxx, maxy].

        Returns:
            search (Search): The search object with the bounding box filter applied.

        Notes:
            The bounding box is transformed into a polygon using the `bbox2polygon` function and
            a geo_shape filter is added to the search object, set to intersect with the specified polygon.
        """
        pass
        # geojson_polygon = {"type": "Polygon", "coordinates": bbox2polygon(*bbox)}
        # search.add_filter(
        #     {
        #         "geometry": {
        #             "$geoIntersects": {
        #                 "$geometry": geojson_polygon,
        #             }
        #         }
        #     }
        # )
        # return search

    @staticmethod
    def apply_intersects_filter(
        search: DuckDBSearchAdapter,
        intersects: Geometry,
    ):
        """Filter search results based on intersecting geometry.

        Args:
            search (Search): The search object to apply the filter to.
            intersects (Geometry): The intersecting geometry, represented as a GeoJSON-like object.

        Returns:
            search (Search): The search object with the intersecting geometry filter applied.

        Notes:
            A geo_shape filter is added to the search object, set to intersect with the specified geometry.
        """
        pass
        # geometry_dict = {"type": intersects.type, "coordinates": intersects.coordinates}
        # search.add_filter(
        #     {"geometry": {"$geoIntersects": {"$geometry": geometry_dict}}}
        # )
        # return search

    @staticmethod
    def apply_stacql_filter(
        search: DuckDBSearchAdapter, op: str, field: str, value: float
    ):
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

    @staticmethod
    def translate_cql2_to_mongo(cql2_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a CQL2 filter dictionary to a MongoDB query.

        This function translates a CQL2 JSON filter into a MongoDB query format. It supports
        various comparison operators, logical operators, and a special handling for spatial
        intersections and the 'in' operator.

        Args:
            cql2_filter: A dictionary representing the CQL2 filter.

        Returns:
            A MongoDB query as a dictionary.
        """
        pass
        # print("CQL2 filter:", cql2_filter)
        # op_mapping = {
        #     ">": "$gt",
        #     ">=": "$gte",
        #     "<": "$lt",
        #     "<=": "$lte",
        #     "=": "$eq",
        #     "!=": "$ne",
        #     "like": "$regex",
        #     "in": "$in",
        # }

        # if cql2_filter["op"] in ["and", "or"]:
        #     mongo_op = f"${cql2_filter['op']}"
        #     return {
        #         mongo_op: [
        #             DatabaseLogic.translate_cql2_to_mongo(arg)
        #             for arg in cql2_filter["args"]
        #         ]
        #     }

        # elif cql2_filter["op"] == "not":
        #     translated_condition = DatabaseLogic.translate_cql2_to_mongo(
        #         cql2_filter["args"][0]
        #     )
        #     return {"$nor": [translated_condition]}

        # elif cql2_filter["op"] == "s_intersects":
        #     geometry = cql2_filter["args"][1]
        #     return {"geometry": {"$geoIntersects": {"$geometry": geometry}}}

        # elif cql2_filter["op"] == "between":
        #     property_name = cql2_filter["args"][0]["property"]

        #     # Use the special mapping directly if available, or construct the path appropriately
        #     if property_name in filter.queryables_mapping:
        #         property_path = filter.queryables_mapping[property_name]
        #     elif property_name not in [
        #         "id",
        #         "collection",
        #     ] and not property_name.startswith("properties."):
        #         property_path = f"properties.{property_name}"
        #     else:
        #         property_path = property_name

        #     lower_bound = cql2_filter["args"][1]
        #     upper_bound = cql2_filter["args"][2]
        #     return {property_path: {"$gte": lower_bound, "$lte": upper_bound}}

        # else:
        #     property_name = cql2_filter["args"][0]["property"]
        #     # Check if the property name is in the special mapping
        #     if property_name in filter.queryables_mapping:
        #         property_path = filter.queryables_mapping[property_name]
        #     elif property_name not in [
        #         "id",
        #         "collection",
        #     ] and not property_name.startswith("properties."):
        #         property_path = f"properties.{property_name}"
        #     else:
        #         property_path = property_name

        #     value = cql2_filter["args"][1]
        #     # Attempt to convert numeric string to float or integer
        #     try:
        #         if "." in value:
        #             value = float(value)
        #         else:
        #             value = int(value)
        #     except (ValueError, TypeError):
        #         pass  # Keep value as is if conversion is not possible
        #     mongo_op = op_mapping.get(cql2_filter["op"])

        #     if mongo_op is None:
        #         raise ValueError(
        #             f"Unsupported operation '{cql2_filter['op']}' in CQL2 filter."
        #         )

        #     if mongo_op == "$regex":
        #         # Replace SQL LIKE wildcards with regex equivalents, handling escaped characters
        #         regex_pattern = re.sub(
        #             r"(?<!\\)%", ".*", value
        #         )  # Replace '%' with '.*', ignoring escaped '\%'
        #         regex_pattern = re.sub(
        #             r"(?<!\\)_", ".", regex_pattern
        #         )  # Replace '_' with '.', ignoring escaped '\_'

        #         # Handle escaped wildcards by reverting them to their literal form
        #         regex_pattern = regex_pattern.replace("\\%", "%").replace("\\_", "_")

        #         # Ensure backslashes are properly escaped for MongoDB regex
        #         regex_pattern = regex_pattern.replace("\\", "\\\\")

        #         return {property_path: {"$regex": regex_pattern, "$options": "i"}}

        #     elif mongo_op == "$in":
        #         if not isinstance(value, list):
        #             raise ValueError(f"Arg {value} is not a list")
        #         return {property_path: {mongo_op: value}}
        #     else:
        #         return {property_path: {mongo_op: value}}

    @staticmethod
    def apply_cql2_filter(
        search_adapter: "DuckDBSearchAdapter", _filter: Optional[Dict[str, Any]]
    ):
        """
        Apply a CQL2 JSON filter to the MongoDB search adapter.

        This method translates a CQL2 JSON filter into MongoDB's query syntax and adds it to the adapter's filters.

        Args:
            search_adapter (DuckDBSearchAdapter): The MongoDB search adapter to which the filter will be applied.
            _filter (Optional[Dict[str, Any]]): The CQL2 filter as a dictionary. If None, no action is taken.

        Returns:
            DuckDBSearchAdapter: The search adapter with the CQL2 filter applied.
        """
        pass
        # if _filter is not None:
        #     mongo_query = DatabaseLogic.translate_cql2_to_mongo(_filter)
        #     search_adapter.add_filter(mongo_query)

        # print("search adapter: ", search_adapter)
        # return search_adapter

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
        pass
        # if not sortby:
        #     return []

        # mongo_sort = []
        # for sort_extension in sortby:
        #     field = sort_extension.field
        #     # Convert the direction enum to a string, then to MongoDB's expected format
        #     direction = 1 if sort_extension.direction.value == "asc" else -1
        #     mongo_sort.append((field, direction))

        # return mongo_sort

    async def execute_search(
        self,
        search: DuckDBSearchAdapter,
        limit: int,
        token: Optional[str],
        sort: Optional[Dict[str, Dict[str, str]]],
        collection_ids: Optional[List[str]],
        ignore_unavailable: bool = True,
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
        client = self.client
        pass
        # db = self.client[DATABASE]
        # collection = db[ITEMS_INDEX]

        # query = {"$and": search.filters} if search and search.filters else {}

        # print("Query: ", query)
        # if collection_ids:
        #     query["collection"] = {"$in": collection_ids}

        # sort_criteria = sort if sort else [("id", 1)]  # Default sort
        # try:
        #     if token:
        #         last_id = decode_token(token)
        #         query["id"] = {"$gt": last_id}

        #     cursor = collection.find(query).sort(sort_criteria).limit(limit + 1)
        #     items = await cursor.to_list(length=limit + 1)

        #     next_token = None
        #     if len(items) > limit:
        #         next_token = base64.urlsafe_b64encode(
        #             str(items[-1]["_id"]).encode()
        #         ).decode()
        #         items = items[:-1]

        #     maybe_count = None
        #     if not token:
        #         maybe_count = await collection.count_documents(query)

        #     return items, maybe_count, next_token
        # except PyMongoError as e:
        #     print(f"Database operation failed: {e}")
        #     raise

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
        # db = self.client[DATABASE]
        # items_collection = db[ITEMS_INDEX]
        # collections_collection = db[COLLECTIONS_INDEX]

        # collection_exists = await collections_collection.count_documents(
        #     {"id": item["collection"]}, limit=1
        # )
        # if not collection_exists:
        #     raise NotFoundError(f"Collection {item['collection']} does not exist")

        # new_item = item.copy()
        # new_item["_id"] = item.get("_id", ObjectId())

        # existing_item = await items_collection.find_one({"_id": new_item["_id"]})
        # if existing_item:
        #     raise ConflictError(f"Item with _id {item['_id']} already exists")

        # await items_collection.insert_one(new_item)
        # item = serialize_doc(item)

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
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]
        # items_collection = db[ITEMS_INDEX]

        # # Check if the collection exists
        # collection_exists = await collections_collection.count_documents(
        #     {"id": item["collection"]}, limit=1
        # )
        # if not collection_exists:
        #     raise NotFoundError(f"Collection {item['collection']} does not exist")

        # # Transform item using item_serializer for MongoDB compatibility
        # mongo_item = self.item_serializer.stac_to_db(item, base_url)

        # if not exist_ok:
        #     existing_item = await items_collection.find_one({"id": mongo_item["id"]})
        #     if existing_item:
        #         raise ConflictError(
        #             f"Item {mongo_item['id']} in collection {mongo_item['collection']} already exists"
        #         )

        # # Return the transformed item ready for insertion
        # return serialize_doc(mongo_item)

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
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]
        # items_collection = db[ITEMS_INDEX]

        # # Check if the collection exists
        # collection_exists = collections_collection.count_documents(
        #     {"id": item["collection"]}, limit=1
        # )
        # if not collection_exists:
        #     raise NotFoundError(f"Collection {item['collection']} does not exist")

        # # Transform item using item_serializer for MongoDB compatibility
        # mongo_item = self.item_serializer.stac_to_db(item, base_url)
        # print("mongo item id: ", mongo_item["id"])
        # if not exist_ok:
        #     existing_item = items_collection.find_one({"id": mongo_item["id"]})
        #     if existing_item:
        #         raise ConflictError(
        #             f"Item {mongo_item['id']} in collection {mongo_item['collection']} already exists"
        #         )

        # # Return the transformed item ready for insertion
        # return serialize_doc(mongo_item)

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
        # db = self.client[DATABASE]
        # items_collection = db[ITEMS_INDEX]

        # try:
        #     # Attempt to delete the item from the collection
        #     result = await items_collection.delete_one({"id": item_id})
        #     if result.deleted_count == 0:
        #         # If no items were deleted, it means the item did not exist
        #         raise NotFoundError(
        #             f"Item {item_id} in collection {collection_id} not found"
        #         )
        # except PyMongoError as e:
        #     # Catch any MongoDB error and re-raise as NotFoundError for consistency with the original function's behavior
        #     raise NotFoundError(
        #         f"Error deleting item {item_id} in collection {collection_id}: {e}"
        #     )

    async def create_collection(self, collection: Collection, refresh: bool = False):
        """Create a single collection document in the database.

        Args:
            collection (Collection): The Collection object to be created.
            refresh (bool, optional): Whether to refresh the index after the creation. Default is False.

        Raises:
            ConflictError: If a Collection with the same id already exists in the database.
        """
        pass
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]

        # # Check if the collection already exists
        # existing_collection = await collections_collection.find_one(
        #     {"id": collection["id"]}
        # )
        # if existing_collection:
        #     raise ConflictError(f"Collection {collection['id']} already exists")

        # try:
        #     # Insert the new collection document into the collections collection
        #     await collections_collection.insert_one(collection)
        # except PyMongoError as e:
        #     # Catch any MongoDB error and raise an appropriate error
        #     print(f"Failed to create collection {collection['id']}: {e}")
        #     raise ConflictError(f"Failed to create collection {collection['id']}: {e}")

        # collection = serialize_doc(collection)

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
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]

        # # Ensure the existing collection exists
        # existing_collection = await self.find_collection(collection_id)
        # if not existing_collection:
        #     raise NotFoundError(f"Collection {collection_id} not found")

        # # Handle changing collection ID
        # if collection_id != collection["id"]:
        #     new_id_exists = await collections_collection.find_one(
        #         {"id": collection["id"]}
        #     )
        #     if new_id_exists:
        #         raise ConflictError(
        #             f"Collection with ID {collection['id']} already exists"
        #         )

        #     items_collection = db[ITEMS_INDEX]
        #     # Update only items related to the old collection ID to the new collection ID
        #     await items_collection.update_many(
        #         {"collection": collection_id},
        #         {"$set": {"collection": collection["id"]}},
        #     )

        #     # Insert the new collection and delete the old one
        #     await collections_collection.insert_one(collection)
        #     await collections_collection.delete_one({"id": collection_id})
        # else:
        #     # Update the existing collection with new data, ensuring not to attempt to update `_id`
        #     await collections_collection.update_one(
        #         {"id": collection_id},
        #         {"$set": {k: v for k, v in collection.items() if k != "_id"}},
        #     )

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
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]
        # items_collection = db[ITEMS_INDEX]

        # # Attempt to delete the collection document
        # collection_result = await collections_collection.delete_one(
        #     {"id": collection_id}
        # )
        # if collection_result.deleted_count == 0:
        #     # Collection not found, raise an error
        #     raise NotFoundError(f"Collection {collection_id} not found")

        # # Successfully found and deleted the collection, now delete its items
        # await items_collection.delete_many({"collection": collection_id})

    async def bulk_async(
        self, collection_id: str, processed_items: List[Item], refresh: bool = False
    ) -> None:
        """Perform a bulk insert of items into the database asynchronously.

        Args:
            self: The instance of the object calling this function.
            collection_id (str): The ID of the collection to which the items belong.
            processed_items (List[Item]): A list of `Item` objects to be inserted into the database.
            refresh (bool): Whether to refresh the index after the bulk insert (default: False).

        Notes:
            This function performs a bulk insert of `processed_items` into the database using the specified `collection_id`. The
            insert is performed asynchronously, and the event loop is used to run the operation in a separate executor. The
            `mk_actions` function is called to generate a list of actions for the bulk insert. If `refresh` is set to True, the
            index is refreshed after the bulk insert. The function does not return any value.
        """
        pass
        # db = self.client[DATABASE]
        # items_collection = db[ITEMS_INDEX]

        # # Prepare the documents for insertion
        # documents = [item.dict(by_alias=True) for item in processed_items]

        # try:
        #     await items_collection.insert_many(documents, ordered=False)
        # except BulkWriteError as e:
        #     # Handle bulk write errors, e.g., due to duplicate keys
        #     raise ConflictError(f"Bulk insert operation failed: {e.details}")

    def bulk_sync(
        self, collection_id: str, processed_items: List[Item], refresh: bool = False
    ) -> None:
        """Perform a bulk insert of items into the database synchronously.

        Args:
            self: The instance of the object calling this function.
            collection_id (str): The ID of the collection to which the items belong.
            processed_items (List[Item]): A list of `Item` objects to be inserted into the database.
            refresh (bool): Whether to refresh the index after the bulk insert (default: False).

        Notes:
            This function performs a bulk insert of `processed_items` into the database using the specified `collection_id`. The
            insert is performed synchronously and blocking, meaning that the function does not return until the insert has
            completed. The `mk_actions` function is called to generate a list of actions for the bulk insert. If `refresh` is set to
            True, the index is refreshed after the bulk insert. The function does not return any value.
        """
        pass
        # db = self.sync_client[DATABASE]
        # items_collection = db[ITEMS_INDEX]

        # # Prepare the documents for insertion
        # documents = [item.dict(by_alias=True) for item in processed_items]

        # try:
        #     items_collection.insert_many(documents, ordered=False)
        # except BulkWriteError as e:
        #     # Handle bulk write errors, e.g., due to duplicate keys
        #     raise ConflictError(f"Bulk insert operation failed: {e.details}")

    async def delete_items(self) -> None:
        """
        Danger. this is only for tests.

        Deletes all items from the 'items' collection in MongoDB.
        """
        pass
        # db = self.client[DATABASE]
        # items_collection = db[ITEMS_INDEX]

        # try:
        #     await items_collection.delete_many({})
        #     print("All items have been deleted.")
        # except Exception as e:
        #     print(f"Error deleting items: {e}")

    async def delete_collections(self) -> None:
        """
        Danger. this is only for tests.

        Deletes all collections from the 'collections' collection in MongoDB.
        """
        pass
        # db = self.client[DATABASE]
        # collections_collection = db[COLLECTIONS_INDEX]

        # try:
        #     await collections_collection.delete_many({})
        #     print("All collections have been deleted.")
        # except Exception as e:
        #     print(f"Error deleting collections: {e}")
