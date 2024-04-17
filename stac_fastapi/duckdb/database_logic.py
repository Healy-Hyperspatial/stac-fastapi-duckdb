"""Database logic."""
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Type, Union

import attr
from fastapi import HTTPException

# from stac_fastapi.core.extensions import filter
# from stac_fastapi.core.utilities import bbox2polygon
from stac_fastapi.core import serializers
from stac_fastapi.extensions.core import SortExtension
from stac_fastapi.types.errors import NotFoundError  # ConflictError
from stac_fastapi.types.stac import Collection, Item

from stac_fastapi.duckdb.config import DuckDBSettings
from stac_fastapi.duckdb.utilities import create_stac_item

logger = logging.getLogger(__name__)

NumType = Union[float, int]


class Geometry(Protocol):  # noqa
    type: str
    coordinates: Any


@attr.s(auto_attribs=True)
class DatabaseLogic:
    """Database logic managing DuckDB connections."""

    settings = DuckDBSettings.get_instance()  # Access the singleton instance
    conn = settings.conn
    stac_file_path: str = os.getenv("STAC_FILE_PATH", "")
    item_serializer: Type[serializers.ItemSerializer] = serializers.ItemSerializer
    collection_serializer: Type[
        serializers.CollectionSerializer
    ] = serializers.CollectionSerializer

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
                raise HTTPException(
                    status_code=404,
                    detail=f"Item {item_id} in collection {collection_id} does not exist.",
                )
            return create_stac_item(df=df, collection_id=collection_id, item_id=item_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
    def apply_datetime_filter(search: dict, datetime_search):
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
        return search

    @staticmethod
    def apply_bbox_filter(search: dict, bbox: List):
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
        return search

    @staticmethod
    def apply_intersects_filter(
        search: dict,
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

    @staticmethod
    def apply_cql2_filter(search_adapter: dict, _filter: Optional[Dict[str, Any]]):
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

    async def execute_search(
        self,
        search: dict,
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
        if not collection_ids:
            raise HTTPException(status_code=400, detail="No collection IDs provided.")
        
        print("SORT: ", sort)

        # Base query construction
        base_query = "SELECT * FROM items"
        count_query = (
            "SELECT COUNT(*) FROM items"  # Query to count total items without limit
        )
        base_params = []  # Parameters for the base query
        count_params = []  # Parameters specifically for the count query
        conditions = []

        # Handle collection_ids if provided
        if collection_ids:
            placeholder = ", ".join("?" * len(collection_ids))
            conditions.append(f"collection IN ({placeholder})")
            base_params.extend(collection_ids)
            count_params.extend(collection_ids)  # Extend count_params similarly

        # # Add conditions from the search dictionary
        # if search:
        #     for key, value in search.items():
        #         conditions.append(f"{key} = ?")
        #         base_params.append(value)
        #         count_params.append(value)  # Add similarly to count_params

        # Combine all conditions into the WHERE clause
        if conditions:
            combined_conditions = " WHERE " + " AND ".join(conditions)
            base_query += combined_conditions
            count_query += combined_conditions

        # Add sorting clause if provided to the base query
        if sort:
            sort_clause = ", ".join(
                f"{field} {direction['order']}" for field, direction in sort.items()
            )
            base_query += f" ORDER BY {sort_clause}"

        # Add LIMIT to the base query
        if limit:
            base_query += " LIMIT ?"
            base_params.append(str(limit))

        try:
            # Execute count query
            total_count = self.conn.execute(count_query, count_params).fetchone()[0]

            # Execute the base query
            df = self.conn.execute(base_query, base_params).df()
            if df.empty and not ignore_unavailable:
                raise HTTPException(
                    status_code=404, detail="No items found in specified collections."
                )

            results = [
                create_stac_item(
                    df=df, collection_id=collection_ids[0], item_id=row["id"]
                )
                for index, row in df.iterrows()
            ]
            return (
                results,
                total_count,
                None,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
