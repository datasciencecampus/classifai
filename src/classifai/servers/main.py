# pylint: disable=C0301
"""This module provides functionality for creating a start a restAPI service.
This allows a user to call the search methods of different VectorStore objects, from
an api-endpoint.

These functions interact with the ClassifAI PackageIndexer modules
VectorStore objects, such that their embed and search methods are exposed on
restAPI endpoints, in a FastAPI restAPI service started with these functions.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated, Literal

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from .. import __version__
from ..exceptions import ConfigurationError, DataValidationError
from ..indexers.dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreReverseSearchInput,
    VectorStoreSearchInput,
)
from ..indexers.main import VectorStore
from .pydantic_models import (
    EmbedRequestSet,
    EmbedResponseBody,
    ReverseSearchRequestSet,
    ReverseSearchResponseBody,
    SearchRequestSet,
    SearchResponseBody,
    convert_embedding_dataframe_to_pydantic_response,
    convert_reverse_search_dataframe_to_pydantic_response,
    convert_search_dataframe_to_pydantic_response,
)


def get_router(vector_stores: list[VectorStore], endpoint_names: list[str]) -> APIRouter:
    """Create and return a `FastAPI.APIRouter` with search endpoints.

    Args:
        vector_stores (list[VectorStore]): A list of `VectorStore` objects, each responsible for handling embedding and search operations for a specific endpoint.
        endpoint_names (list[str]): A list of endpoint names corresponding to the vector stores.

    Returns:
        (APIRouter): Router with intialized search endpoints

    Raises:
        `DataValidationError`: Raised if the input parameters are invalid.
        `ConfigurationError`: Raised if one or more of the `vector_stores` are invalid.

    """
    # ---- Validate startup args -> DataValidationError / ConfigurationError
    if not isinstance(vector_stores, list) or not isinstance(endpoint_names, list):
        raise DataValidationError(
            "vector_stores and endpoint_names must be lists.",
            context={
                "vector_stores_type": type(vector_stores).__name__,
                "endpoint_names_type": type(endpoint_names).__name__,
            },
        )

    if len(vector_stores) != len(endpoint_names):
        raise DataValidationError(
            "The number of vector stores must match the number of endpoint names.",
            context={"n_vector_stores": len(vector_stores), "n_endpoint_names": len(endpoint_names)},
        )

    if any(not isinstance(x, str) or not x.strip() for x in endpoint_names):
        raise DataValidationError(
            "All endpoint_names must be non-empty strings.",
            context={"endpoint_names": endpoint_names},
        )

    if len(set(endpoint_names)) != len(endpoint_names):
        raise DataValidationError(
            "endpoint_names must be unique.",
            context={"endpoint_names": endpoint_names},
        )

    for i, vs in enumerate(vector_stores):
        if not isinstance(vs, VectorStore):
            raise ConfigurationError(
                "vector_store must be an instance of the VectorStore class.",
                context={"index": i, "vector_store_type": type(vs).__name__},
            )

    logging.info("Starting ClassifAI Router")
    router = APIRouter()
    vector_stores_dict: dict[str, VectorStore] = dict(zip(endpoint_names, vector_stores, strict=True))
    make_endpoints(router, vector_stores_dict)

    @router.get("/", description="UI accessibility", tags=["docs"])
    def docs():
        """Redirect users to the API documentation page.

        Returns:
            (RedirectResponse): A response object that redirects the user to the `/docs` page.
        """
        start_page = RedirectResponse(url="/docs")
        return start_page

    return router


def get_server(vector_stores: list[VectorStore], endpoint_names: list[str]) -> FastAPI:
    """Create and return a `FastAPI` server with search endpoints.

    Args:
        vector_stores (list[VectorStore]): A list of `VectorStore` objects, each responsible for handling embedding and search operations for a specific endpoint.
        endpoint_names (list[str]): A list of endpoint names corresponding to the `VectorStore`s to be exposed.

    Returns:
        (FastAPI): Server with intialized search endpoints
    """
    logging.info("Generating ClassifAI API")

    openapi_tags = [
        {"name": endpoint, "description": f"Endpoints for the {endpoint} VectorStore"} for endpoint in endpoint_names
    ]
    app = FastAPI(
        title="ClassifAI API Server",
        description="This is the Classifai FastAPI server",
        openapi_tags=openapi_tags,
        version=__version__,
    )

    router = get_router(vector_stores, endpoint_names)
    app.include_router(router)
    return app


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def run_server(  # noqa: PLR0913
    vector_stores: list[VectorStore],
    endpoint_names: list[str],
    port: int = 8000,
    host_ip: str = "127.0.0.1",
    log_level: LogLevel | str = LogLevel.WARNING,
    demo_mode: bool = False,
):
    """Create and run a `FastAPI` server with search endpoints.

    Args:
        vector_stores (list[VectorStore]): A list of `VectorStore` objects, each responsible for handling embedding and search operations for a specific endpoint.
        endpoint_names (list[str]): A list of endpoint names corresponding to the `VectorStore`s to be exposed.
        port (int): [optional] The port on which the API server will run. Defaults to 8000.
        host_ip (str): [optional] The ip address that the api server runs on. Defaults to 127.0.0.1, note: default 127.0.0.1 exposes to connections from the same machine only, to expose for external connections use 0.0.0.0.
        log_level (str): [optional] The level of logs for the uvicorn server.
        demo_mode (bool): [optional] Flag to show demo server info (Updates the openapi docs to show info indicating server is an api demo).

    Raises:
        `DataValidationError`: Raised if the input parameters are invalid, e.g. `port` value is out of bounds.
    """
    logging.info("Starting ClassifAI API")

    MAX_PORT, MIN_PORT = 65535, 1
    if not isinstance(port, int) or port < MIN_PORT or port > MAX_PORT:
        raise DataValidationError(
            "port must be an integer between 1 and 65535.",
            context={"port": port},
        )

    if log_level not in LogLevel:
        raise DataValidationError(
            f"Invalid log level '{log_level}'. Must be one of: {list(LogLevel)}", context={"log_level": log_level}
        )

    app = get_server(vector_stores, endpoint_names)

    if demo_mode:
        _set_demo_defaults(app)

    uvicorn.run(app, port=port, log_level=log_level, host=host_ip)


def _set_demo_defaults(app: FastAPI):
    app.title = "ClassifAI API Demo Server"
    app.description = "This is a demo of the ClassifAI server module"


def make_endpoints(main_router: APIRouter | FastAPI, vector_stores_dict: dict[str, VectorStore]):
    """Create and register the different endpoints to your app.

    Args:
        main_router (APIRouter | FastAPI): The FastAPI application instance.
        vector_stores_dict (dict[str, VectorStore]): The name of the endpoint to be created.
    """
    for name, store in vector_stores_dict.items():
        sub_router = APIRouter(
            prefix=f"/{name}",
            tags=[name],
        )
        logging.info("Registering endpoints for: %s", name)

        _create_search_endpoint(sub_router, name, store)
        _create_embedding_endpoint(sub_router, name, store)
        _create_reverse_search_endpoint(sub_router, name, store)

        main_router.include_router(sub_router)


def _create_embedding_endpoint(router: APIRouter | FastAPI, endpoint_name: str, vector_store: VectorStore):
    """Create and register an embedding endpoint for a specific `VectorStore`.

    Args:
        router (APIRouter | FastAPI): The `FastAPI` application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The vector store object responsible for generating embeddings.

    The created endpoint accepts POST requests with input data, generates embeddings
    for the provided documents, and returns the results in a structured format.
    """

    @router.post(
        "/embed",
        summary=f"{endpoint_name} Embedding Endpoint",
        description=f"Endpoint to call the `{endpoint_name}` `VectorStore.embed` method",
    )
    async def embedding_endpoint(data: EmbedRequestSet) -> EmbedResponseBody:
        input_ids = [x.id for x in data.entries]
        input_texts = [x.text for x in data.entries]

        # Creat the input dataclass object and pass it to the vectorstore to get results.
        input_data = VectorStoreEmbedInput({"id": input_ids, "text": input_texts})
        output_data = vector_store.embed(input_data)

        # post processing of the Vectorstore output åobject
        formatted_result = convert_embedding_dataframe_to_pydantic_response(output_data)

        return formatted_result


def _create_search_endpoint(router: APIRouter | FastAPI, endpoint_name: str, vector_store: VectorStore):
    """Create and register a search endpoint for a specific `VectorStore`.

    Args:
        router (APIRouter | FastAPI): The `FastAPI` application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The `VectorStore` object responsible for performing search operations.

    The created endpoint accepts POST requests with input data and a query parameter
    specifying the number of results to return. It performs a search operation using
    the vector store and returns the results in a structured format.
    """

    @router.post(
        "/search",
        summary=f"{endpoint_name} Search Endpoint",
        description=f"Endpoint to call the `{endpoint_name}` `VectorStore.search` method",
    )
    async def search_endpoint(
        data: SearchRequestSet,
        n_results: Annotated[
            int,
            Query(
                description="The number of knowledgebase results to return per input query.",
                ge=1,  # Ensure at least one result is returned
            ),
        ] = 10,
    ) -> SearchResponseBody:
        # Creat the input dataclass object and pass it to the vectorstore to get results.
        input_ids = [x.id for x in data.entries]
        queries = [x.query for x in data.entries]

        # Creat the input dataclass object and pass it to the vectorstore to get results.
        input_data = VectorStoreSearchInput({"id": input_ids, "query": queries})
        output_data = vector_store.search(query=input_data, n_results=n_results)

        # post processing of the Vectorstore output åobject
        formatted_result = convert_search_dataframe_to_pydantic_response(
            df=output_data,
            meta_data=vector_store.meta_data,
        )

        return formatted_result


def _create_reverse_search_endpoint(router: APIRouter | FastAPI, endpoint_name: str, vector_store: VectorStore):
    """Create and register a reverse_search endpoint for a specific vector store.

    Args:
        router (APIRouter | FastAPI): The `FastAPI` application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The `VectorStore` object responsible for performing search operations.

    The created endpoint accepts POST requests with input data and a query parameter
    specifying the number of results to return. It performs a reverse search operation using
    the vector store and returns the results in a structured format.
    """

    @router.post(
        "/reverse_search",
        summary=f"{endpoint_name} Reverse Search Endpoint",
        description=f"Endpoint to call the `{endpoint_name}` `VectorStore.reverse_search` method",
    )
    def reverse_search_endpoint(
        data: ReverseSearchRequestSet,
        max_n_results: Annotated[
            int | Literal[-1],
            Query(description="The max number of results to return, set to -1 to return all results."),
        ] = 100,
        partial_match: Annotated[
            bool, Query(description="Flag to use partial `starts_with` matching for queries")
        ] = False,
    ) -> ReverseSearchResponseBody:
        # Enforce the ≥1 rule manually, only when not -1
        if max_n_results != -1 and max_n_results < 1:
            raise HTTPException(422, "max_n_results must be -1 or >= 1")

        # Creat the input dataclass object and pass it to the vectorstore to get results.
        input_ids = [x.id for x in data.entries]
        queries = [x.doc_label for x in data.entries]

        # Creat the input dataclass object and pass it to the vectorstore to get results.
        input_data = VectorStoreReverseSearchInput({"id": input_ids, "doc_label": queries})
        output_data = vector_store.reverse_search(input_data, max_n_results=max_n_results, partial_match=partial_match)

        # post processing of the Vectorstore output object
        formatted_result = convert_reverse_search_dataframe_to_pydantic_response(
            df=output_data,
            meta_data=vector_store.meta_data,
        )
        return formatted_result
