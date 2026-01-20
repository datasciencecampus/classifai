# pylint: disable=C0301
"""This module provides functionality for creating a start a restAPI service which
allows a user to call the search methods of different VectorStore objects, from
an api-endpoint.

These functions interact with the ClassifAI PackageIndexer modules
VectorStore objects, such that their embed and search methods are exposed on
restAPI endpoints, in a FastAPI restAPI service started with these functions.
"""

import logging
from typing import Annotated

import uvicorn

# New imports
from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse

from ..indexers.dataclasses import VectorStoreEmbedInput, VectorStoreReverseSearchInput, VectorStoreSearchInput
from .pydantic_models import (
    ClassifaiData,
    EmbeddingsList,
    EmbeddingsResponseBody,
    RevClassifaiData,
    convert_dataframe_to_pydantic_response,
    convert_dataframe_to_reverse_search_pydantic_response,
)


def setup_api(vector_stores, endpoint_names, hooks: list[dict] | None = None):  # noqa: C901,PLR0915,PLR0912
    """Initialize the FastAPI application with dynamically created endpoints.
    This function dynamically registers embedding and search endpoints for each provided
    vector store and endpoint name. It also sets up a default route to redirect users to
    the API documentation page.

    Args:
        vector_stores (list): A list of vector store objects, each responsible for handling
                              embedding and search operations for a specific endpoint.
        endpoint_names (list): A list of endpoint names corresponding to the vector stores.
        port (int, optional): The port on which the API server will run. Defaults to 8000.
        hooks (list[dict], optional): A list of hook dictionaries (one per VectorStore)
                                      for additional configurations. Defaults to [{}].
                                      Hook dictionaries should be in the format
                                        {
                                            "embed": {
                                                "decorators": [callable_1, callable_2, ...],
                                                "pre_endpt": callable,
                                                "post_endpt": callable,
                                            },
                                            ...
                                        }
    """
    if len(vector_stores) != len(endpoint_names):
        raise ValueError("The number of vector stores must match the number of endpoint names.")

    logging.info("Starting ClassifAI API")

    endpoint_index_map = {x: i for i, x in enumerate(endpoint_names)}

    app = FastAPI()

    def create_embedding_endpoint(app, endpoint_name, vector_store):
        """Create and register an embedding endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for generating embeddings.
            hooks (dict, optional): A dictionary of hooks for additional configurations.
                                    Defaults to {}.

        The created endpoint accepts POST requests with input data, generates embeddings
        for the provided documents, and returns the results in a structured format.
        """

        async def embedding_endpoint(request: Request, data: ClassifaiData):
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

        return embedding_endpoint

    def create_search_endpoint(app, endpoint_name, vector_store):
        """Create and register a search endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for performing search operations.

        The created endpoint accepts POST requests with input data and a query parameter
        specifying the number of results to return. It performs a search operation using
        the vector store and returns the results in a structured format.
        """

        async def search_endpoint(
            request: Request,
            data: ClassifaiData,
            n_results: Annotated[
                int,
                Query(
                    description="The number of knowledgebase results to return per input query.",
                    ge=1,  # Ensure at least one result is returned
                ),
            ] = 10,
        ):
            input_ids = [x.id for x in data.entries]
            queries = [x.description for x in data.entries]

            input_data = VectorStoreSearchInput({"id": input_ids, "query": queries})
            output_data = vector_store.search(query=input_data, n_results=n_results)

            ##post processing of the Vectorstore outputobject
            formatted_result = convert_dataframe_to_pydantic_response(
                df=output_data,
                meta_data=vector_stores[endpoint_index_map[endpoint_name]].meta_data,
            )

            return formatted_result

        return search_endpoint

    def create_reverse_search_endpoint(app, endpoint_name, vector_store):
        """Create and register a reverse_search endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for performing search operations.

        The created endpoint accepts POST requests with input data and a query parameter
        specifying the number of results to return. It performs a reverse search operation using
        the vector store and returns the results in a structured format.
        """

        def reverse_search_endpoint(
            request: Request,
            data: RevClassifaiData,
            n_results: Annotated[
                int,
                Query(
                    description="The max number of results to return.",
                ),
            ] = 100,
        ):
            input_ids = [x.id for x in data.entries]
            queries = [x.code for x in data.entries]

            input_data = VectorStoreReverseSearchInput({"id": input_ids, "doc_id": queries})
            output_data = vector_store.reverse_search(input_data, n_results=n_results)

            formatted_result = convert_dataframe_to_reverse_search_pydantic_response(
                df=output_data,
                meta_data=vector_stores[endpoint_index_map[endpoint_name]].meta_data,
            )
            return formatted_result

        return reverse_search_endpoint

    for endpoint_name, vector_store, hooks_dict in zip(endpoint_names, vector_stores, hooks, strict=True):
        logging.info("Registering endpoints for: %s", endpoint_name)
        embedding_endpt = create_embedding_endpoint(app, endpoint_name, vector_store)
        if hooks_dict is not None:
            if hooks_dict.get("embed"):
                for decorator in hooks_dict["embed"].get("decorators", []):
                    embedding_endpt = decorator(embedding_endpt)
                embedding_endpt = app.post(
                    f"/{endpoint_name}/embed", description=f"{endpoint_name} embedding endpoint"
                )(embedding_endpt)
                if pre_endpt := hooks_dict["embed"].get("pre_endpt"):
                    embedding_endpt = pre_endpt(embedding_endpt)
                if post_endpt := hooks_dict["embed"].get("post_endpt"):
                    embedding_endpt = post_endpt(embedding_endpt)
            search_endpt = create_search_endpoint(app, endpoint_name, vector_store)
            if hooks_dict.get("search"):
                for decorator in hooks_dict["search"].get("decorators", []):
                    search_endpt = decorator(search_endpt)
                search_endpt = app.post(f"/{endpoint_name}/search", description=f"{endpoint_name} search endpoint")(
                    search_endpt
                )
                if pre_endpt := hooks_dict["search"].get("pre_endpt"):
                    search_endpt = pre_endpt(search_endpt)
                if post_endpt := hooks_dict["search"].get("post_endpt"):
                    search_endpt = post_endpt(search_endpt)
            reverse_search_endpt = create_reverse_search_endpoint(app, endpoint_name, vector_store)
            if hooks_dict.get("reverse_search"):
                for decorator in hooks_dict["reverse_search"].get("decorators", []):
                    reverse_search_endpt = decorator(reverse_search_endpt)
                reverse_search_endpt = app.post(
                    f"/{endpoint_name}/reverse_search", description=f"{endpoint_name} reverse query endpoint"
                )(reverse_search_endpt)
                if pre_endpt := hooks_dict["reverse_search"].get("pre_endpt"):
                    reverse_search_endpt = pre_endpt(reverse_search_endpt)
                if post_endpt := hooks_dict["reverse_search"].get("post_endpt"):
                    reverse_search_endpt = post_endpt(reverse_search_endpt)

    @app.get("/", description="UI accessibility")
    def docs():
        """Redirect users to the API documentation page.

        Returns:
            RedirectResponse: A response object that redirects the user to the `/docs` page.
        """
        start_page = RedirectResponse(url="/docs")
        return start_page

    return app


def start_api(app, port: int = 8000):
    """Start the FastAPI application using Uvicorn."""
    uvicorn.run(app, port=port, log_level="info")
