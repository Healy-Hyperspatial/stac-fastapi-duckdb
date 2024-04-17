"""Database utility functions."""
from datetime import date, datetime

import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.wkb import loads


def decode_geometry(int_list):
    """Convert the integer list to bytes."""
    geom_bytes = bytes(int_list)

    # Use Shapely to decode the WKB format
    geometry_object = loads(geom_bytes)

    # Convert to GeoJSON format
    return mapping(geometry_object)


def convert_type(value):
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
                return pd.to_datetime(
                    value
                ).isoformat()  # Ensure datetime64 types are also converted
            elif np.issubdtype(value.dtype, np.complex):
                return str(value.item())  # Convert complex numbers to string
            else:
                return value.item()  # Last resort, convert directly
        elif isinstance(value, (datetime, date)):
            return (
                value.isoformat()
            )  # Additional handling for native Python datetime types
        return value
    except Exception as e:
        print(f"Failed conversion for value {value} of type {type(value)}: {e}")
        return None  # Or choose to return a default value or placeholder


def create_stac_item(df, item_id, collection_id):
    """Create stac item."""
    # Decode the WKB geometry to a shapely object
    geom_int_list = df.at[0, "geometry"]
    geojson_geometry = decode_geometry(geom_int_list)

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": convert_type(df.at[0, "stac_extensions"]),
        "id": item_id,
        "bbox": df.at[0, "bbox"],
        "collection": collection_id,
        "properties": {},
        "geometry": geojson_geometry,
        "assets": df.at[0, "assets"],
        "links": [],
    }

    # Dynamically populate properties with improved error handling
    for column in df.columns:
        if column not in [
            "id",
            "geometry",
            "assets",
            "links",
            "type",
            "bbox",
            "stac_version",
            "stac_extensions",
        ]:
            converted_value = convert_type(df.at[0, column])
            item["properties"][column] = (
                converted_value if converted_value is not None else None
            )

    return item
