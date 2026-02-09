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

# New imports
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse

from ..indexers.dataclasses import VectorStoreEmbedInput, VectorStoreReverseSearchInput, VectorStoreSearchInput
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


class ClassifAIServer(FastAPI):
    def __init__(self, vectorstores: list[VectorStore], endpointnames: list[str]):
        self.endpoint_names = endpointnames

        if len(vectorstores) != len(self.endpoint_names):
            raise ValueError("The number of vector stores must match the number of endpoint names.")

        self.vector_stores: dict[str, VectorStore] = dict(zip(endpointnames, vectorstores, strict=False))

    def create_embedding_endpoint(self, endpoint_name: str):
        """Create and register an embedding endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for generating embeddings.

        The created endpoint accepts POST requests with input data, generates embeddings
        for the provided documents, and returns the results in a structured format.
        """

        @self.post(f"/{endpoint_name}/embed", description=f"{endpoint_name} embedding endpoint")
        async def embedding_endpoint(data: ClassifaiData) -> EmbeddingsResponseBody:
            input_ids = [x.id for x in data.entries]
            documents = [x.description for x in data.entries]

            input_data = VectorStoreEmbedInput({"id": input_ids, "text": documents})

            output_data = self.vector_stores[endpoint_name].embed(input_data)

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

    def create_search_endpoint(self, endpoint_name: str):
        """Create and register a search endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for performing search operations.

        The created endpoint accepts POST requests with input data and a query parameter
        specifying the number of results to return. It performs a search operation using
        the vector store and returns the results in a structured format.
        """

        @self.post(f"/{endpoint_name}/search", description=f"{endpoint_name} search endpoint")
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
            output_data = self.vector_stores[endpoint_name].search(query=input_data, n_results=n_results)

            ##post processing of the Vectorstore outputobject
            formatted_result = convert_dataframe_to_pydantic_response(
                df=output_data,
                meta_data=self.vector_stores[endpoint_name].meta_data,
            )

            return formatted_result

    def create_reverse_search_endpoint(self, endpoint_name: str):
        """Create and register a reverse_search endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for performing search operations.

        The created endpoint accepts POST requests with input data and a query parameter
        specifying the number of results to return. It performs a reverse search operation using
        the vector store and returns the results in a structured format.
        """

        @self.post(f"/{endpoint_name}/reverse_search", description=f"{endpoint_name} reverse query endpoint")
        def reverse_search_endpoint(
            data: RevClassifaiData,
            n_results: Annotated[
                int,
                Query(
                    description="The max number of results to return.",
                ),
            ] = 100,
        ) -> RevResultsResponseBody:
            input_ids = [x.id for x in data.entries]
            queries = [x.code for x in data.entries]

            input_data = VectorStoreReverseSearchInput({"id": input_ids, "doc_id": queries})
            output_data = self.vector_stores[endpoint_name].reverse_search(input_data, n_results=n_results)

            formatted_result = convert_dataframe_to_reverse_search_pydantic_response(
                df=output_data,
                meta_data=self.vector_stores[endpoint_name].meta_data,
            )
            return formatted_result

    def start(self, port: int = 8000):
        """Run the server."""
        logging.info("Starting ClassifAI API")

        uvicorn.run(self, port=port, log_level="info")

    def make_endpoints(self):
        """Setup the different endpoints."""
        for endpoint_name in self.endpoint_names:
            logging.info("Registering endpoints for: %s", endpoint_name)
            self.create_embedding_endpoint(endpoint_name)
            self.create_search_endpoint(endpoint_name)
            self.create_reverse_search_endpoint(endpoint_name)

        @self.get("/", description="UI accessibility")
        def docs():
            """Redirect users to the API documentation page.

            Returns:
                RedirectResponse: A response object that redirects the user to the `/docs` page.
            """
            start_page = RedirectResponse(url="/docs")
            return start_page
