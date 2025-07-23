"""Functions to initiate the API endpoints."""

import logging
from typing import Annotated

import uvicorn

# New imports
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse

from .pydantic_models import (
    ClassifaiData,
    EmbeddingsResponseBody,
    ResultsResponseBody,
    convert_dataframe_to_pydantic_response,
)


def start_api(vector_stores, endpoint_names, port=8000):
    """Start the API with the given vector stores and endpoint names."""
    logging.info("Starting ClassifAI API")

    endpoint_index_map = {x: i for i, x in enumerate(endpoint_names)}

    app = FastAPI()

    def create_embedding_endpoint(app, endpoint_name, vector_store):
        """Helper function to create embedding endpoint."""

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
        """Helper function to create search endpoint."""

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

    for endpoint_name, vector_store in zip(endpoint_names, vector_stores):
        logging.info(f"Registering endpoints for: {endpoint_name}")
        create_embedding_endpoint(app, endpoint_name, vector_store)
        create_search_endpoint(app, endpoint_name, vector_store)

    @app.get("/", description="UI accessibility")
    def docs():
        """Access default page: docs UI."""

        start_page = RedirectResponse(url="/docs")
        return start_page

    uvicorn.run(app, port=8000, log_level="info")
