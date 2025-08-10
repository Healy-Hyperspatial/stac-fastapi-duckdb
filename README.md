# stac-fastapi-duckdb

<!-- markdownlint-disable MD033 MD041 -->

<p align="left">
  <img src="https://github.com/radiantearth/stac-site/raw/master/images/logo/stac-030-long.png" width=600>
</p>


### DuckDB backend for the stac-fastapi project built on top of the [sfeos](https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch) core api library. 

### Working so far:
- GET /collections   
- GET /collections/{collection_id}
- GET /collections/{collection_id}/items
- GET /collections/{collection_id}/items?limit=1
- GET /collections/{collection_id}/items/{item_id}

- Example: http://localhost:8085/collections

### To install from PyPI (not implemented yet):

```shell
pip install stac_fastapi.duckdb
```

### For changes, see the [Changelog](CHANGELOG.md)


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

## Configuration

This backend queries GeoParquet directly (local paths, HTTP, or S3) using per-request DuckDB connections. Configure which Parquet belongs to which STAC collection via environment variables.

- __PARQUET_URLS_JSON__ (required): JSON mapping of `collection_id -> url_or_path`.
  - Local example: `{"my_collection": "file:///data/my.parquet"}`
  - S3 example (public bucket): `{"landsat": "s3://public-bucket/path/landsat.parquet"}`
  - You may use globs: `{"cog": "s3://bucket/prefix/*.parquet"}`

- __HTTP_CACHE_PATH__ (optional, default `/tmp/duckdb_http_cache`): Where DuckDB caches HTTP/S3 metadata (and object cache if available) to reduce network round trips.

- __Threads__ (optional): set the `threads` field in code or expose as env to control DuckDB execution threads.

Example (bash):

```bash
export PARQUET_URLS_JSON='{"demo": "file:///data/demo.parquet", "landsat": "s3://usgs-public/landsat/*.parquet"}'
export HTTP_CACHE_PATH=/tmp/duckdb_http_cache
```

Notes:

- For S3 without auth, ensure the bucket/objects are public.
- For remote HTTP/S3, the service enables DuckDB httpfs and spatial extensions per request and uses metadata/object cache when available.
- One GeoParquet corresponds to one STAC collection id.

