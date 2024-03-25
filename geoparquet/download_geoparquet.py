import sys
import geopandas
import planetary_computer
import pystac_client

if len(sys.argv) != 2:
    print("Usage: python download_geoparquet.py <collection_id>")
    sys.exit(1)

# Use the collection ID passed as a command-line argument
collection_id = sys.argv[1]

# Set up the STAC client and access the specified collection
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/",
    modifier=planetary_computer.sign_inplace,
)

try:
    collection = catalog.get_collection(collection_id)
    asset = collection.assets["geoparquet-items"]
except KeyError:
    print(f"Collection '{collection_id}' not found or does not have a 'geoparquet-items' asset.")
    sys.exit(1)

# Read the GeoParquet data into a GeoDataFrame
df = geopandas.read_parquet(
    asset.href, storage_options=asset.extra_fields["table:storage_options"]
)

# Display the first few rows of the GeoDataFrame
print(df.head())

# Optionally, save the GeoDataFrame to a local GeoParquet file
df.to_parquet(f"{collection_id}.parquet")
