# stac-fastapi-duckdb

<!-- markdownlint-disable MD033 MD041 -->

<p align="left">
  <img src="https://github.com/radiantearth/stac-site/raw/master/images/logo/stac-030-long.png" width=600>
</p>

### DuckDB backend for the stac-fastapi project built on top of the [sfeos](https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch) core api library.

## Quick Start with Docker

The easiest way to get started is using Docker and the provided Makefile:

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-org/stac-fastapi-duckdb.git
   cd stac-fastapi-duckdb
   ```

2. **Build and start the Docker container**:
   ```bash
   make up
   ```
   This will:
   - Build the Docker image
   - Start the STAC API server on http://localhost:8085
   - Mount the `stac_collections` directory into the container

3. **Access the API**:
   - Browse collections: http://localhost:8085/collections
   - View collection items: http://localhost:8085/collections/io-lulc-9-class/items
   - Get a specific item: http://localhost:8085/collections/io-lulc-9-class/items/{item_id}

4. **Other useful commands**:
   ```bash
   # Run in detached mode (background)
   make up-d
   
   # View logs
   make logs
   
   # Stop the container
   make down
   ```

## API Endpoints

The following STAC API endpoints are implemented:
- `GET /collections` - List all collections
- `GET /collections/{collection_id}` - Get a specific collection
- `GET /collections/{collection_id}/items` - Get items in a collection
- `GET /collections/{collection_id}/items/{item_id}` - Get a specific item

## Configuration

### Environment Variables

- `PARQUET_URLS_JSON` (required): JSON object mapping collection IDs to Parquet file paths/URLs
  - Local file example: `{"io-lulc-9-class": "file:///app/stac_collections/io-lulc-9-class/io-lulc-9-class.parquet"}`
  - S3 example: `{"landsat": "s3://public-bucket/path/landsat.parquet"}`
  - When running with Docker, use container paths (e.g., `/app/stac_collections/...`)

- `HTTP_CACHE_PATH` (optional, default: `/tmp/duckdb_http_cache`): 
  Directory where DuckDB caches HTTP/S3 metadata to reduce network round trips

- `STAC_FILE_PATH` (optional, default: `/app/stac_collections`):
  Directory containing STAC collection JSON files

- `ENABLE_DIRECT_RESPONSE` (optional, default: `false`):
  Enable direct response handling

- `RAISE_ON_BULK_ERROR` (optional, default: `false`):
  Whether to raise exceptions on bulk operation errors

## Development

### Pre-commit

Install [pre-commit](https://pre-commit.com/#install).

Prior to commit, run:

```shell
pre-commit run --all-files
```

## Build stac-fastapi.duckdb backend

```shell
docker compose build
```
  
## Running DuckDB API on localhost:8085

```shell
docker compose up
```

