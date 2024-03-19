"""download demo geoparquet script."""

import geopandas
import planetary_computer
import pystac_client

# Set up the STAC client and access the specified collection
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/",
    modifier=planetary_computer.sign_inplace,
)
collection_id = "io-lulc-9-class"
collection = catalog.get_collection(collection_id)
asset = collection.assets["geoparquet-items"]

# Read the GeoParquet data into a GeoDataFrame
df = geopandas.read_parquet(
    asset.href, storage_options=asset.extra_fields["table:storage_options"]
)

# Display the first few rows of the GeoDataFrame
print(df.head())

# Optionally, save the GeoDataFrame to a local GeoParquet file
df.to_parquet(f"{collection_id}.parquet")
