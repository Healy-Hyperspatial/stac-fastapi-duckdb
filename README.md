# stac-fastapi-duckdb

<!-- markdownlint-disable MD033 MD041 -->

<p align="left">
  <img src="https://github.com/radiantearth/stac-site/raw/master/images/logo/stac-030-long.png" width=600>
  <p align="left"><b>DuckDB backend for the stac-fastapi project.</b></p>
</p>


## DuckDB backend for the stac-fastapi project built on top of the [sfeos](https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch) core api library. 

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
