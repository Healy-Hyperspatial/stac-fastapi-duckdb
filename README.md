# stac-fastapi-duckdb

<!-- markdownlint-disable MD033 MD041 -->

<p align="left">
  <img src="https://github.com/radiantearth/stac-site/raw/master/images/logo/stac-030-long.png" width=600>
</p>

### DuckDB backend for the stac-fastapi project built on top of the [sfeos](https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch) core api library.

## Technologies

This project is built on the following technologies: STAC, stac-fastapi, SFEOS core, FastAPI, DuckDB, Python

<p align="left">
  <a href="https://stacspec.org/"><img src="https://raw.githubusercontent.com/stac-utils/stac-fastapi-elasticsearch-opensearch/refs/heads/main/assets/STAC-01.png" alt="STAC" height="100" hspace="10"></a>
  <a href="https://www.python.org/"><img src="https://raw.githubusercontent.com/stac-utils/stac-fastapi-elasticsearch-opensearch/refs/heads/main/assets/python.png" alt="Python" height="80" hspace="10"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://raw.githubusercontent.com/stac-utils/stac-fastapi-elasticsearch-opensearch/refs/heads/main/assets/fastapi.svg" alt="FastAPI" height="80" hspace="10"></a>
  <a href="https://duckdb.org/"><img src="https://raw.githubusercontent.com/Healy-Hyperspatial/stac-fastapi-duckdb/refs/heads/main/assets/duckdb-icon-logo-png.png" alt="DuckDB" height="80" hspace="10"></a>
  <a href="https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch"><img src="https://raw.githubusercontent.com/Healy-Hyperspatial/stac-fastapi-mongo/refs/heads/main/assets/sfeos-bw.png" alt="stac-fastapi-core" height="83" hspace="10"></a>
</p>

## Table of Contents

- [Quick Start with Docker](#quick-start-with-docker)
- [Usage](#usage)
  - [Supported Query Parameters](#supported-query-parameters)
  - [Example Queries](#example-queries)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Pre-commit](#pre-commit)

## Quick Start with Docker

The easiest way to get started is using Docker and the provided Makefile:

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/Healy-Hyperspatial/stac-fastapi-duckdb.git
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
- `GET /collections/{collection_id}/items` - Get items in a collection with filtering support (bbox, datetime)
- `GET /collections/{collection_id}/items/{item_id}` - Get a specific item
- `POST /search` - Search across collections with advanced filtering (bbox, datetime, etc.)

Both the GET items endpoint and POST search endpoint support bbox filtering. The GET endpoint accepts bbox as a comma-separated string query parameter, while the POST endpoint accepts bbox as either a comma-separated string or an array of numbers in the request body.

### Supported Query Parameters

#### Spatial Filtering
- `bbox` - Filter items by bounding box in format `west,south,east,north`
  - Example: `bbox=-66,-16,-60,-8`
  - Uses DuckDB's spatial extension with ST_Intersects for efficient filtering

#### Temporal Filtering
- `datetime` - Filter items by temporal extent using RFC3339 datetime strings:
  - Single datetime: `datetime=2022-01-01T00:00:00Z`
  - Date range: `datetime=2022-01-01T00:00:00Z/2023-01-01T00:00:00Z`
  - Open-ended ranges: `datetime=2022-01-01T00:00:00Z/..` or `datetime=../2023-01-01T00:00:00Z`

#### Other Parameters
- `limit` - Maximum number of items to return (default: 10)
- `bbox` - Spatial bounding box filter: `bbox=west,south,east,north`
- `ids` - Filter by specific item IDs: `ids=item1,item2,item3`

### Example Queries

```bash
# Get items from a specific time range
curl "http://localhost:8085/collections/io-lulc-9-class/items?datetime=2019-01-01T00:00:00Z/2023-01-01T00:00:00Z&limit=5"

# Get items with spatial filtering (bbox format: west,south,east,north)
curl "http://localhost:8085/collections/io-lulc-9-class/items?bbox=-66,-16,-60,-8"

# Get items with both spatial and temporal filters
curl "http://localhost:8085/collections/io-lulc-9-class/items?bbox=-66,-16,-60,-8&datetime=2020-01-01T00:00:00Z/2022-01-01T00:00:00Z"

# Search across all collections
curl -X POST "http://localhost:8085/search" \
  -H "Content-Type: application/json" \
  -d '{"datetime": "2019-01-01T00:00:00Z/2023-01-01T00:00:00Z", "limit": 10}'

# Search with bbox filter in POST request
curl -X POST "http://localhost:8085/search" \
  -H "Content-Type: application/json" \
  -d '{"bbox": [-66, -16, -60, -8], "limit": 10}'
```

## Configuration

### Environment Variables

- `PARQUET_URLS_JSON` (required): JSON object mapping collection IDs to Parquet file paths/URLs
  - Local file example: `{"io-lulc-9-class": "file:///app/stac_collections/io-lulc-9-class/io-lulc-9-class.parquet"}`
  - S3 example: `{"landsat": "s3://public-bucket/path/landsat.parquet"}`
  - When running with Docker, use container paths (e.g., `/app/stac_collections/...`)

- `STAC_FILE_PATH` (optional, default: `/app/stac_collections`):
  Directory containing STAC collection JSON files

## Development

### Running Tests

The project includes a Makefile with commands to run tests:

```bash
# Build and run tests in Docker
make test-build

# Run tests in existing Docker container
make test
```

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

