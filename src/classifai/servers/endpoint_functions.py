from typing import Annotated

from fastapi import Query

from .pydantic_models import (
    ClassifaiData,
    EmbeddingsList,
    EmbeddingsResponseBody,
    RagResponseBody,
    ResultsResponseBody,
    RevClassifaiData,
    RevResultsResponseBody,
    convert_dataframe_to_pydantic_response,
    convert_dataframe_to_reverse_search_pydantic_response,
)


def create_embedding_endpoint(app, endpoint_name, vector_store):
    """Create and register an embedding endpoint for a specific vector store.

    Args:
        app (FastAPI): The FastAPI application instance.
        endpoint_name (str): The name of the endpoint to be created.
        vector_store: The vector store object responsible for generating embeddings.

    The created endpoint accepts POST requests with input data, generates embeddings
    for the provided documents, and returns the results in a structured format.
    """

    @app.post(f"/{endpoint_name}/embed", description=f"{endpoint_name} embedding endpoint")
    async def embedding_endpoint(data: ClassifaiData) -> EmbeddingsResponseBody:
        input_ids = [x.id for x in data.entries]
        documents = [x.description for x in data.entries]

        embeddings = vector_store.embed(documents)

        returnable = []
        for idx, desc, embed in zip(input_ids, documents, embeddings, strict=True):
            returnable.append(
                EmbeddingsList(
                    idx=idx,
                    description=desc,
                    embedding=embed.tolist(),
                )
            )
        return EmbeddingsResponseBody(data=returnable)


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

    @app.post(f"/{endpoint_name}/search", description=f"{endpoint_name} search endpoint")
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

        query_result = vector_store.search(query=queries, ids=input_ids, n_results=n_results)
        ##post processing of the pandas dataframe
        formatted_result = convert_dataframe_to_pydantic_response(
            df=query_result,
            meta_data=vector_store.meta_data,
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

    @app.post(f"/{endpoint_name}/reverse_search", description=f"{endpoint_name} reverse query endpoint")
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

        reverse_query_result = vector_store.reverse_search(query=queries, ids=input_ids, n_results=n_results)

        formatted_result = convert_dataframe_to_reverse_search_pydantic_response(
            df=reverse_query_result,
            meta_data=vector_store.meta_data,
            ids=input_ids,
        )
        return formatted_result


def create_rag_endpoint(app, endpoint_name, rag_agent) -> RagResponseBody:
    """Create and register a search endpoint for a specific vector store.

    Args:
        app (FastAPI): The FastAPI application instance.
        endpoint_name (str): The name of the endpoint to be created.
        rag_agent: The rag_agent object responsible for performing search and generation operations

    The created endpoint accepts POST requests with input string which is a prompt to a Generative
    Rag agent. It performs a search operation using a vectorstore and then generates a response
    according to the preset task type returning a [agent_response, ranking] dict in a structured format.
    """

    @app.post(
        f"/{endpoint_name}/prompt",
        description=f"{endpoint_name} RAG agent endpoint, where a user can prompt the agent",
    )
    async def rag_endpoint(prompt: str) -> RagResponseBody:
        agent_response = rag_agent.transform(prompt=prompt)

        formatted_ranking = convert_dataframe_to_pydantic_response(
            df=agent_response["ranking"],
            meta_data=rag_agent.vectorStore.meta_data,
        )

        return RagResponseBody(agent_response=agent_response["text"], ranking=formatted_ranking)
