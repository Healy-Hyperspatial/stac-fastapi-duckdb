SHELL := /bin/bash

.DEFAULT_GOAL := help

PROJECT_ROOT := $(abspath .)
STAC_DIR := $(PROJECT_ROOT)/stac_collections
DEMO_PARQUET := $(STAC_DIR)/io-lulc-9-class/io-lulc-9-class.parquet
PARQUET_URL := file:///app/stac_collections/io-lulc-9-class/io-lulc-9-class.parquet


help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*##"} {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## Build docker image
	docker compose build

up: ## Run docker compose in foreground with demo env
	STAC_FILE_PATH="$(STAC_DIR)" PARQUET_URLS_JSON='{"io-lulc-9-class":"$(PARQUET_URL)"}' docker compose up

up-d: ## Run docker compose detached with demo env
	STAC_FILE_PATH="$(STAC_DIR)" PARQUET_URLS_JSON='{"io-lulc-9-class":"$(PARQUET_URL)"}' docker compose up -d

down: ## Stop containers
	docker compose down

logs: ## Tail logs
	docker compose logs -f

DOCKER_ENV = STAC_FILE_PATH="$(STAC_DIR)" \
	PARQUET_URLS_JSON='{"io-lulc-9-class":"$(PARQUET_URL)"}'

test: ## Run tests in docker
	$(DOCKER_ENV) docker compose exec app-duckdb pytest tests/ -v

test-build: ## Build and run tests in docker
	$(DOCKER_ENV) docker compose build
	$(DOCKER_ENV) docker compose up -d
	@echo "Waiting for containers to be ready..."
	@sleep 5
	$(DOCKER_ENV) docker compose exec app-duckdb pytest tests/ -v
	$(DOCKER_ENV) docker compose down

restart: down up-d ## Restart detached

demo-url: ## Print demo PARQUET_URLS_JSON
	@echo '{"io-lulc-9-class":"$(PARQUET_URL)"}'

# --- Demo helpers ---
demo: ## Build image and run demo (detached)
	$(MAKE) build
	$(MAKE) up-d
	@echo "\nDemo running at: http://localhost:8085"
	@echo "Try:"
	@echo "  - http://localhost:8085/collections"
	@echo "  - http://localhost:8085/collections/io-lulc-9-class"
	@echo "  - http://localhost:8085/collections/io-lulc-9-class/items?limit=1"

demo-down: ## Stop the demo containers
	$(MAKE) down

demo-logs: ## Tail demo logs
	$(MAKE) logs
