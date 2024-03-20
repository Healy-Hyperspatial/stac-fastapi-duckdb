# stac-fastapi-duckdb

## DuckDB backend for the stac-fastapi project built on top of the [sfeos](https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch) core api library. 

### This is a work in progress and is not functional yet. If you have any ideas related to this project please feel free to share. Please feel free to contribute as well. I am not 100% sure on how to proceed as I am new to DuckDB and this project will be different than stac-fastapi-pgstac or stac-fastapi-elasticsearch, becasue DuckDB is different.    

To install from PyPI (not implemented yet):

```shell
pip install stac_fastapi.duckdb
```

#### For changes, see the [Changelog](CHANGELOG.md)


## Development Environment Setup

To install the classes in your local Python env, run:

```shell
pip install -e .[dev]
```


### Pre-commit

Install [pre-commit](https://pre-commit.com/#install).

Prior to commit, run:

```shell
pre-commit run --all-files
```

## Build stac-fastapi.duckdb backend

```shell
```
  
## Running DuckDB API on localhost:8084

```shell
```

To create a new Collection:

```shell
curl -X "POST" "http://localhost:8084/collections" \
     -H 'Content-Type: application/json; charset=utf-8' \
     -d $'{
  "id": "my_collection"
}'
```


## Collection pagination

The collections route handles optional `limit` and `token` parameters. The `links` field that is
returned from the `/collections` route contains a `next` link with the token that can be used to 
get the next page of results.
   
```shell
curl -X "GET" "http://localhost:8084/collections?limit=1&token=example_token"
```

## Testing

```shell
make test
```


## Ingest sample data

```shell
make ingest
```
