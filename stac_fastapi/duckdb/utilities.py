"""Database utility functions."""
from datetime import date, datetime
from logging import getLogger

import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.wkb import loads

logger = getLogger(__name__)


def decode_geometry(int_list):
    """Convert the integer list to bytes."""
    geom_bytes = bytes(int_list)

    # Use Shapely to decode the WKB format
    geometry_object = loads(geom_bytes)

    # Convert to GeoJSON format
    return mapping(geometry_object)


def convert_type(value):
    """Convert unsupported numpy types to native Python types for JSON serialization.

    Handles various types including:
    - NumPy arrays and scalars
    - Pandas NA/NaN values
    - Datetime objects
    - Projection-related fields (proj:bbox, proj:shape, proj:transform)
    """
    try:
        # Handle None/NA values first
        if value is None or (hasattr(value, "size") and value.size == 0):
            return None

        # Handle pandas NA/NaT
        if (
            hasattr(value, "isna") and value.isna().any()
            if hasattr(value, "any")
            else False
        ):
            return None

        # Handle NumPy arrays and scalars first
        if isinstance(value, np.ndarray):
            # For numeric arrays, convert directly to list
            if value.dtype.kind in "iufc":  # integer, unsigned, float, complex
                return value.tolist()
            # For object arrays, convert each element
            elif value.dtype.kind == "O":
                return [convert_type(x) for x in value]
            # For other array types, convert to list
            else:
                return value.tolist()

        # Handle NumPy scalar types
        if isinstance(value, np.generic):
            if np.issubdtype(value.dtype, np.floating):
                return float(value.item())
            elif np.issubdtype(value.dtype, np.integer):
                return int(value.item())
            elif np.issubdtype(value.dtype, np.bool_):
                return bool(value.item())
            elif np.issubdtype(value.dtype, np.datetime64):
                return pd.to_datetime(value).isoformat()
            elif np.issubdtype(value.dtype, np.complex):
                return str(value.item())
            return value.item()

        # Handle datetime objects
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()

        # Handle pandas Series
        if hasattr(value, "to_dict"):
            value = value.to_dict()

        # Handle lists, tuples, and other sequences (but not strings/bytes)
        if isinstance(value, (list, tuple, np.ma.MaskedArray)) and not isinstance(
            value, (str, bytes)
        ):
            # Convert masked arrays to regular lists first
            if hasattr(value, "filled"):
                value = value.filled(None)
            return [convert_type(x) for x in value if x is not None]

        # Handle dictionaries
        if isinstance(value, dict):
            return {
                str(k): convert_type(v)
                for k, v in value.items()
                if v is not None and (not hasattr(v, "size") or v.size > 0)
            }

        # For any other type, try direct conversion or return as is
        try:
            # Try to convert to native Python type
            if hasattr(value, "item"):
                return value.item()
            return value
        except Exception:
            # If conversion fails, return string representation
            return str(value)

    except Exception as e:
        logger.debug(
            f"Type conversion warning for value of type {type(value)}: {str(e)}"
        )
        return None  # Return None for any conversion failures


def create_stac_item(df, item_id, collection_id):
    """
    Create a STAC item from a DataFrame row.

    Args:
        df: DataFrame containing the item data
        item_id: ID of the STAC item
        collection_id: ID of the parent collection

    Returns:
        dict: A STAC item as a dictionary

    Raises:
        Exception: If required fields are missing or invalid
    """
    try:
        # Get row data safely
        if len(df) == 0:
            raise ValueError("No data in DataFrame")

        row = df.iloc[0] if hasattr(df, "iloc") else df

        # Only geometry is truly required for a valid STAC item
        if "geometry" not in df.columns:
            raise ValueError("Missing required field: geometry")

        # Decode the WKB geometry to a GeoJSON geometry
        try:
            geom_int_list = row.get("geometry")
            if geom_int_list is None:
                raise ValueError("Geometry is None")
            geojson_geometry = decode_geometry(geom_int_list)
        except Exception as e:
            raise ValueError(f"Failed to decode geometry: {str(e)}")

        # Get bbox - don't warn if missing, it's optional in STAC
        bbox = None
        if "bbox" in row:
            try:
                bbox_raw = row["bbox"]
                if bbox_raw is not None:
                    bbox = convert_type(bbox_raw)
                    # Only validate if we got a value
                    if bbox is not None and not (
                        isinstance(bbox, (list, tuple)) and len(bbox) in [4, 6]
                    ):
                        bbox = None
            except Exception:
                pass  # Silently ignore bbox conversion errors

        # Define special fields that should be handled separately
        special_fields = {
            "id",
            "geometry",
            "assets",
            "links",
            "type",
            "bbox",
            "stac_version",
            "stac_extensions",
            "collection",
        }

        # Define projection fields that need special handling
        projection_fields = {
            "proj:epsg",
            "proj:geometry",
            "proj:bbox",
            "proj:transform",
            "proj:shape",
        }

        # Initialize properties dictionary
        properties = {}

        # Process all fields in the row
        for column in row.index if hasattr(row, "index") else row.keys():
            try:
                # Skip special fields and None values
                if column in special_fields or row[column] is None:
                    continue

                # Convert the value using our type converter
                value = row[column]

                # Special handling for projection fields
                if column in projection_fields:
                    try:
                        converted = convert_type(value)
                        if converted is not None:
                            properties[column] = converted
                        continue
                    except Exception:
                        continue  # Skip if conversion fails

                # Handle regular properties
                try:
                    converted = convert_type(value)
                    if converted is not None:
                        properties[column] = converted
                except Exception:
                    continue  # Skip if conversion fails

            except Exception as e:
                logger.debug(f"Skipping field {column}: {str(e)}")
                continue

        # Build the base item
        item = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": convert_type(row.get("stac_extensions", [])),
            "id": item_id,
            "collection": collection_id,
            "properties": properties,
            "geometry": geojson_geometry,
            "assets": convert_type(row.get("assets", {})),
            "links": convert_type(row.get("links", [])),
        }

        # Add bbox if we have it
        if bbox is not None:
            item["bbox"] = bbox

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
                try:
                    value = row.get(column)
                    # Handle arrays and scalars safely to avoid ambiguous truth value warnings
                    if value is not None:
                        # For arrays, check if they have any elements; for scalars, check if not NaN
                        if hasattr(value, '__len__') and hasattr(value, 'size'):
                            # NumPy array or similar - check if it has elements
                            if value.size > 0:
                                item["properties"][column] = convert_type(value)
                        elif pd.notna(value):
                            # Scalar value - use pd.notna
                            item["properties"][column] = convert_type(value)
                except Exception as e:
                    print(f"Warning: Could not process property '{column}': {str(e)}")
                    continue

        return item

    except Exception as e:
        # Log the full error for debugging
        print(f"Error creating STAC item {item_id}: {str(e)}")
        print(f"Row data: {dict(row) if 'row' in locals() else 'No row data'}")
        raise  # Re-raise the exception to be handled by the caller
