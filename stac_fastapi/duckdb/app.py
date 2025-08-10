"""FastAPI application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from stac_fastapi.api.app import StacApi
from stac_fastapi.api.models import create_get_request_model, create_post_request_model
from stac_fastapi.core.core import CoreClient
from stac_fastapi.core.extensions import QueryExtension
from stac_fastapi.core.route_dependencies import get_route_dependencies
from stac_fastapi.core.session import Session
from stac_fastapi.extensions.core import FieldsExtension, FilterExtension, SortExtension
from stac_fastapi.extensions.core.filter import FilterConformanceClasses
from stac_fastapi.sfeos_helpers.filter import EsAsyncBaseFiltersClient

from stac_fastapi.duckdb.config import DuckDBSettings
from stac_fastapi.duckdb.database_logic import DatabaseLogic

settings = DuckDBSettings()
session = Session.create_from_settings(settings)

database_logic = DatabaseLogic()

# Initialize extensions
filter_extension = FilterExtension(
    client=EsAsyncBaseFiltersClient(database=database_logic)
)
filter_extension.conformance_classes.append(
    FilterConformanceClasses.ADVANCED_COMPARISON_OPERATORS
)

extensions = [
    FieldsExtension(),
    QueryExtension(),
    SortExtension(),
    filter_extension,
]

database_logic.extensions = [type(ext).__name__ for ext in extensions]

post_request_model = create_post_request_model(extensions)

# Create app config dictionary
app_config = {
    "title": os.getenv("STAC_FASTAPI_TITLE", "stac-fastapi-duckdb"),
    "description": os.getenv("STAC_FASTAPI_DESCRIPTION", "stac-fastapi-duckdb"),
    "api_version": os.getenv("STAC_FASTAPI_VERSION", "0.0.1"),
    "settings": settings,
    "extensions": extensions,
    "client": CoreClient(
        database=database_logic,
        session=session,
        post_request_model=post_request_model,
        landing_page_id=os.getenv("STAC_FASTAPI_LANDING_PAGE_ID", "stac-fastapi"),
    ),
    "search_get_request_model": create_get_request_model(extensions),
    "search_post_request_model": post_request_model,
    "route_dependencies": get_route_dependencies(),
}

# Initialize StacApi with config
try:
    api = StacApi(**app_config)
except Exception as e:
    print(f"Error initializing StacApi: {e}")
    raise

# Add lifespan context


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for FastAPI app."""
    # Add any initialization needed for DuckDB
    yield


# Create FastAPI app
app = api.app
app.router.lifespan_context = lifespan
app.root_path = os.getenv("STAC_FASTAPI_ROOT_PATH", "")


def run() -> None:
    """Run app from command line using uvicorn if available."""
    try:
        import uvicorn

        print("host: ", settings.app_host)
        print("port: ", settings.app_port)
        uvicorn.run(
            "stac_fastapi.duckdb.app:app",
            host=settings.app_host,
            port=settings.app_port,
            log_level="info",
            reload=settings.reload,
        )
    except ImportError:
        raise RuntimeError("Uvicorn must be installed in order to use command")


if __name__ == "__main__":
    run()


def create_handler(app):
    """Create a handler to use with AWS Lambda if mangum available."""
    try:
        from mangum import Mangum

        return Mangum(app)
    except ImportError:
        return None


handler = create_handler(app)
