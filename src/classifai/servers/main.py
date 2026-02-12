# pylint: disable=C0301
"""This module provides functionality for creating a start a restAPI service which
allows a user to call the search methods of different VectorStore objects, from
an api-endpoint.

These functions interact with the ClassifAI PackageIndexer modules
VectorStore objects, such that their embed and search methods are exposed on
restAPI endpoints, in a FastAPI restAPI service started with these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifai.indexers import VectorStore

import logging
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRouter

from ..exceptions import ConfigurationError, DataValidationError
from ..indexers.dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreReverseSearchInput,
    VectorStoreSearchInput,
)
from ..indexers.main import VectorStore
from .pydantic_models import (
    ClassifaiData,
    EmbeddingsList,
    EmbeddingsResponseBody,
    ResultsResponseBody,
    RevClassifaiData,
    RevResultsResponseBody,
    convert_dataframe_to_pydantic_response,
    convert_dataframe_to_reverse_search_pydantic_response,
)


def get_router(vector_stores: list[VectorStore], endpoint_names: list[str]) -> APIRouter:
    """Create and return a FastAPI router with search endpoints.

    Args:
        vector_stores (list[VectorStore]): A list of vector store objects, each responsible for handling embedding and search operations for a specific endpoint.
        endpoint_names (list[str]): A list of endpoint names corresponding to the vector stores.
        port (int, optional): The port on which the API server will run. Defaults to 8000.

    Returns:
        APIRouter: Router with intialized search endpoints    Raises:
        DataValidationError: If the input parameters are invalid.
        ConfigurationError: If a vector store is missing required methods.

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

    logging.info("Starting ClassifAI Router")
    router = APIRouter()
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

    vector_stores_dict: dict[str, VectorStore] = dict(zip(endpoint_names, vector_stores, strict=True))
    make_endpoints(router, vector_stores_dict)

    @router.get("/", description="UI accessibility")
    def docs():
        """Redirect users to the API documentation page.

        Returns:
            RedirectResponse: A response object that redirects the user to the `/docs` page.
        """
        start_page = RedirectResponse(url="/docs")
        return start_page

    return router


def get_server(vector_stores: list[VectorStore], endpoint_names: list[str]) -> FastAPI:
    """Create and return a FastAPI server with search endpoints.

    Args:
        vector_stores (list[VectorStore]): A list of vector store objects, each responsible for handling embedding and search operations for a specific endpoint.
        endpoint_names (list[str]): A list of endpoint names corresponding to the vector stores.
        port (int, optional): The port on which the API server will run. Defaults to 8000.

    Returns:
        FastAPI: Server with intialized search endpoints
    """
    logging.info("Generating ClassifAI API")

    app = FastAPI(title="ClassifAI Demo Server", description="This is a demo server of the ClassifAI server")
    router = get_router(vector_stores, endpoint_names)
    app.include_router(router)
    return app


def run_server(vector_stores: list[VectorStore], endpoint_names: list[str], port: int = 8000):
    """Create and run a FastAPI server with search endpoints.

    Args:
        vector_stores (list[VectorStore]): A list of vector store objects, each responsible for handling embedding and search operations for a specific endpoint.
        endpoint_names (list[str]): A list of endpoint names corresponding to the vector stores.
        port (int, optional): The port on which the API server will run. Defaults to 8000.
    """
    logging.info("Starting ClassifAI API")

    MAX_PORT, MIN_PORT = 65535, 1
    if not isinstance(port, int) or port < MIN_PORT or port > MAX_PORT:
        raise DataValidationError(
            "port must be an integer between 1 and 65535.",
            context={"port": port},
        )

    app = get_server(vector_stores, endpoint_names)
    uvicorn.run(app, port=port, log_level="info")


def make_endpoints(router: APIRouter | FastAPI, vector_stores_dict: dict[str, VectorStore]):
    """Create and register the different endpoints to your app.

    Args:
        router (APIRouter | FastAPI): The FastAPI application instance.
        vector_stores_dict (dict[str, VectorStore]): The name of the endpoint to be created.
    """
    for endpoint_name, vector_store in vector_stores_dict.items():
        logging.info("Registering endpoints for: %s", endpoint_name)
        create_embedding_endpoint(endpoint_name, vector_store)
        create_search_endpoint(endpoint_name, vector_store)
        create_reverse_search_endpoint(endpoint_name, vector_store)


def create_embedding_endpoint(router: APIRouter | FastAPI, endpoint_name: str, vector_store: VectorStore):
    """Create and register an embedding endpoint for a specific vector store.

    Args:
        router (APIRouter | FastAPI): The FastAPI application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The vector store object responsible for generating embeddings.

    The created endpoint accepts POST requests with input data, generates embeddings
    for the provided documents, and returns the results in a structured format.
    """

    @router.post(f"/{endpoint_name}/embed", description=f"{endpoint_name} embedding endpoint")
    async def embedding_endpoint(data: ClassifaiData) -> EmbeddingsResponseBody:
        input_ids = [x.id for x in data.entries]
        documents = [x.description for x in data.entries]

        input_data = VectorStoreEmbedInput({"id": input_ids, "text": documents})

        output_data = vector_store.embed(input_data)

        returnable = []
        for _, row in output_data.iterrows():
            returnable.append(
                EmbeddingsList(
                    idx=row["id"],
                    description=row["text"],
                    embedding=row["embedding"].tolist(),  # Convert numpy array to list
                )
            )
        return EmbeddingsResponseBody(data=returnable)


def create_search_endpoint(router: APIRouter | FastAPI, endpoint_name: str, vector_store: VectorStore):
    """Create and register a search endpoint for a specific vector store.

    Args:
        router (APIRouter | FastAPI): The FastAPI application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The vector store object responsible for performing search operations.

    The created endpoint accepts POST requests with input data and a query parameter
    specifying the number of results to return. It performs a search operation using
    the vector store and returns the results in a structured format.
    """

    @router.post(f"/{endpoint_name}/search", description=f"{endpoint_name} search endpoint")
    async def search_endpoint(
        data: ClassifaiData,
        n_results: Annotated[
            int,
            Query(
                description="The number of knowledgebase results to return per input query.",
                ge=1,  # Ensure at least one result is returned
            ),
        ] = 10,
    ) -> ResultsResponseBody:
        input_ids = [x.id for x in data.entries]
        queries = [x.description for x in data.entries]

        input_data = VectorStoreSearchInput({"id": input_ids, "query": queries})
        output_data = vector_store.search(query=input_data, n_results=n_results)

        ##post processing of the Vectorstore outputobject
        formatted_result = convert_dataframe_to_pydantic_response(
            df=output_data,
            meta_data=vector_store.meta_data,
        )

        return formatted_result


def create_reverse_search_endpoint(router: APIRouter | FastAPI, endpoint_name: str, vector_store: VectorStore):
    """Create and register a reverse_search endpoint for a specific vector store.

    Args:
        router (APIRouter | FastAPI): The FastAPI application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The vector store object responsible for performing search operations.

    The created endpoint accepts POST requests with input data and a query parameter
    specifying the number of results to return. It performs a reverse search operation using
    the vector store and returns the results in a structured format.
    """

    @router.post(f"/{endpoint_name}/reverse_search", description=f"{endpoint_name} reverse query endpoint")
    def reverse_search_endpoint(
        data: RevClassifaiData,
        n_results: Annotated[
            int,
            Query(description="The max number of results to return.", ge=1),
        ] = 100,
    ) -> RevResultsResponseBody:
        input_ids = [x.id for x in data.entries]
        queries = [x.code for x in data.entries]

        input_data = VectorStoreReverseSearchInput({"id": input_ids, "doc_id": queries})
        output_data = vector_store.reverse_search(input_data, n_results=n_results)

        formatted_result = convert_dataframe_to_reverse_search_pydantic_response(
            df=output_data,
            meta_data=vector_store.meta_data,
        )
        return formatted_result
