# pylint: disable=C0301
"""This module provides functionality for creating a restAPI service which
allows a user to call the search methods of different VectorStore objects, from
designated API endpoints.

These functions interact with the ClassifAI Indexer module
VectorStore objects, such that their embed and search methods are exposed on
restAPI endpoints, in a FastAPI restAPI service started with these functions.
"""

import logging

import uvicorn

# New imports
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from ..generators.base import GeneratorBase
from ..indexers import VectorStore
from .endpoint_functions import (
    create_embedding_endpoint,
    create_rag_endpoint,
    create_reverse_search_endpoint,
    create_search_endpoint,
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

    app = FastAPI()

    for endpoint_name, vector_store in zip(endpoint_names, vector_stores, strict=True):
        logging.info("Registering endpoints for: %s", endpoint_name)

        # Check if the object is an instance of a GeneratorBase subclass or VectorStore and load the corresponding endpoints
        if isinstance(vector_store, GeneratorBase):
            create_rag_endpoint(app, endpoint_name, vector_store)
        elif isinstance(vector_store, VectorStore):
            create_embedding_endpoint(app, endpoint_name, vector_store)
            create_search_endpoint(app, endpoint_name, vector_store)
            create_reverse_search_endpoint(app, endpoint_name, vector_store)
        else:
            raise TypeError(
                f"Unsupported object type for endpoint '{endpoint_name}': {type(vector_store)}. "
                "Expected VectorStore or GeneratorBase instance."
            )

    @app.get("/", description="UI accessibility")
    def docs():
        """Redirect users to the API documentation page.

        Returns:
            RedirectResponse: A response object that redirects the user to the `/docs` page.
        """
        start_page = RedirectResponse(url="/docs")
        return start_page

    uvicorn.run(app, port=port, log_level="info")
