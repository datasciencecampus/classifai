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
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse

from .pydantic_models import (
    ClassifaiData,
    RevClassifaiData,
    ResultsResponseBody,
    RevResultsResponseBody,
    EmbeddingsResponseBody,
    convert_dataframe_to_pydantic_response,
    convert_dataframe_to_reverse_search_pydantic_response
)


def start_api(vector_stores, endpoint_names, port=8000):
    """Initialize and start the FastAPI application with dynamically created endpoints.
    This function dynamically registers embedding and search endpoints for each provided
    vector store and endpoint name. It also sets up a default route to redirect users to
    the API documentation page.

    Args:
        vector_stores (list): A list of vector store objects, each responsible for handling
                              embedding and search operations for a specific endpoint.
        endpoint_names (list): A list of endpoint names corresponding to the vector stores.
        port (int, optional): The port on which the API server will run. Defaults to 8000.


    """
    logging.info("Starting ClassifAI API")

    endpoint_index_map = {x: i for i, x in enumerate(endpoint_names)}

    app = FastAPI()

    def create_embedding_endpoint(app, endpoint_name, vector_store):
        """Create and register an embedding endpoint for a specific vector store.

        Args:
            app (FastAPI): The FastAPI application instance.
            endpoint_name (str): The name of the endpoint to be created.
            vector_store: The vector store object responsible for generating embeddings.

        The created endpoint accepts POST requests with input data, generates embeddings
        for the provided documents, and returns the results in a structured format.
        """

        @app.post(
            f"/{endpoint_name}/embed", description=f"{endpoint_name} embedding endpoint"
        )
        async def embedding_endpoint(data: ClassifaiData) -> EmbeddingsResponseBody:
            input_ids = [x.id for x in data.entries]
            documents = [x.description for x in data.entries]

            embeddings = vector_store.embed(documents)

            returnable = []
            for i in range(len(input_ids)):
                returnable.append(
                    {
                        "idx": input_ids[i],
                        "description": documents[i],
                        "embedding": embeddings[i, :].tolist(),
                    }
                )
            return {"data": returnable}

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

        @app.post(
            f"/{endpoint_name}/search", description=f"{endpoint_name} search endpoint"
        )
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

            query_result = vector_store.search(
                query=queries, ids=input_ids, n_results=n_results
            )
            ##post processing of the pandas dataframe
            formatted_result = convert_dataframe_to_pydantic_response(
                df=query_result,
                meta_data=vector_stores[endpoint_index_map[endpoint_name]].meta_data,
            )

            return formatted_result

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
        @app.post(
                f"/{endpoint_name}/reverse_search", description=f"{endpoint_name} reverse query endpoint"
        )
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

            reverse_query_result = vector_store.reverse_search( query=queries, ids=input_ids, n_results=n_results)

            formatted_result = convert_dataframe_to_reverse_search_pydantic_response(df=reverse_query_result,
                meta_data=vector_stores[endpoint_index_map[endpoint_name]].meta_data,
                ids=input_ids)
            return formatted_result

    for endpoint_name, vector_store in zip(endpoint_names, vector_stores):
        logging.info("Registering endpoints for: %s", endpoint_name)
        create_embedding_endpoint(app, endpoint_name, vector_store)
        create_search_endpoint(app, endpoint_name, vector_store)
        create_reverse_search_endpoint(app,endpoint_name, vector_store)
        
    @app.get("/", description="UI accessibility")
    def docs():
        """Redirect users to the API documentation page.

        Returns:
            RedirectResponse: A response object that redirects the user to the `/docs` page.
        """
        start_page = RedirectResponse(url="/docs")
        return start_page

    uvicorn.run(app, port=8000, log_level="info")
