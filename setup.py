"""stac_fastapi: duckdb module."""

from setuptools import find_namespace_packages, setup

with open("README.md") as f:
    desc = f.read()

install_requires = [
    "duckdb==0.10.1" "stac-fastapi.core==2.0.0",
    "uvicorn",
    "starlette",
]

extra_reqs = {
    "dev": [
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "pre-commit",
        "requests",
        "ciso8601",
        "httpx",
    ],
    "docs": ["mkdocs", "mkdocs-material", "pdocs"],
    "server": ["uvicorn[standard]==0.19.0"],
}

setup(
    name="stac-fastapi.duckdb",
    version="0.1.0",
    description="DuckDB stac-fastapi backend.",
    long_description=desc,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    url="https://github.com/Healy-Hyperspatial/stac-fastapi-duckdb",
    license="MIT",
    packages=find_namespace_packages(),
    zip_safe=False,
    install_requires=install_requires,
    extras_require=extra_reqs,
    entry_points={
        "console_scripts": ["stac-fastapi-duckdb=stac_fastapi.duckdb.app:run"]
    },
)
