"""Functions to initiate the API endpoints."""

import sys
import logging
from typing import Annotated

# New imports
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from google.cloud import storage

from ..fast_api.google_configurations.config import Config

from ..fast_api.deduplication.condense import create_deduplicated_response
from ..fast_api.deduplication.scorers import naive_scorer

from ..fast_api.embedder import ParquetNumpyVectorStore as VectorStore
from ..fast_api.embedder import embed_as_array

from ..fast_api.pydantic_models import ClassifaiData, ResultsResponseBody, EmbeddingsResponseBody

import numpy as np

import uvicorn

app = FastAPI(
    title="ONS ClassifAI API",
    description=("Experimental - " + "For illustrative purposes only."),
    summary="""Experimental API facilitating user-access to ClassifAI product.""",
    version="0.0.1",
    contact={
        "name": "ONS Data Science Campus",
        "email": "dsc.projects@ons.gov.uk",
    },
)

load_dotenv(find_dotenv())

def setup_app():
    global config, vector_store
    config = Config("API")
    config.setup_logging()
    if not config.validate():
        logging.error("Invalid configuration. Exiting.")
        import sys
        sys.exit(1)

    vector_store = VectorStore.from_gcs_bucket(
            client=storage.Client(),
            bucket_name=config.bucket_name,
            local_dir=config.db_dir,
            prefix="soc_parquet",
            force_refresh=True,
    )
    return config, vector_store

@app.post("/soc", description="SOC programmatic endpoint")
def soc(
    data: ClassifaiData,
    n_results: Annotated[
        int,
        Query(
            description="The number of results to return per SOC row.",
        ),
    ] = 20,
) -> ResultsResponseBody:
    """Label input data using SOC programmatic endpoint.

    Parameters
    ----------
    data: ClassifaiData
        user provided JSON array, contains many ClassifaiEntry json objects which are {'id': str, 'description': str}
    n_results: int
        The number of results to return per query

    Returns
    -------
    ResultsResponseBody : ResultsResponseBody :: Modelled as a Pydantic response object which provides example json and type checking
        Dictionary (Pydantic Object) of top n closest codes to input roles.
    """

    input_ids = [x.id for x in data.entries]
    documents = [x.description for x in data.entries]

    query_embeddings = embed_as_array(documents, config.embedding_api_key)
    
    query_result = vector_store.query(
        query_embeddings, ids=input_ids, k=n_results
    )
    processed_result = vector_store.create_json_array_response(query_result)

    # format the deduplicated data response based on the original formatted data
    deduplicated_result = create_deduplicated_response(
        processed_result, naive_scorer
    )

    # return both data and deduplicated data
    return {
        "data": processed_result,
        "deduplicated_data": deduplicated_result,
    }

@app.post("/embed", description="embeddings programmatic endpoint")
def embed(data: ClassifaiData,
) -> EmbeddingsResponseBody :
    """Get the embeddings from a programmatic endpoint.

    Parameters
    ----------
    data: ClassifaiData
        user provided JSON array, contains many ClassifaiEntry json objects which are {'id': str, 'description': str}
    Returns
    -------
    EmbeddingsResponseBody : EmbeddingsResponseBody :: Modelled as a Pydantic response object
        Dictionary (Pydantic Object) of embeddings for each of the input roles.
    """
    # Do type checking, then build pydantic
    
    input_ids = [x.id for x in data.entries]
    documents = [x.description for x in data.entries]

    query_embeddings = embed_as_array(documents, config.embedding_api_key)
    description_labels = vector_store.knowledgebase['description'].to_list()
    formatted_embeddings_package = vector_store.create_embeddings_json_array_response(query_embeddings, documents, input_ids, description_labels)
    #print(vector_store.knowledgebase.sample(n=10))

    return formatted_embeddings_package

'''
@app.post("/score", description="scoring programmatic endpoint")
def score(query_embeddings: list[float],
) -> list[float] :
    """Get the embeddings from a programmatic endpoint.

    Parameters
    ----------
    query_embeddings: list[float]
        user provided query embeddings array
    Returns
    -------
    List of scores for each of query embeddings.
    """
    
    query_embeddings = np.array(query_embeddings)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings)
    document_embeddings = vector_store.knowledgebase["embeddings"].to_numpy().T
    scores = query_embeddings @ document_embeddings
    return scores.tolist()
''';

@app.get("/", description="UI accessibility")
def docs():
    """Access default page: docs UI."""

    start_page = RedirectResponse(url="/docs")
    return start_page

def run_app():

    #config.setup_logging()
    #if not config.validate():
    #    print('here')
    #    logging.error("Invalid configuration. Exiting.")
    #    import sys
    #    sys.exit(1)

    #vector_store = VectorStore.from_gcs_bucket(
    #    client=storage.Client(),
    #    bucket_name=config.bucket_name,
    #    local_dir=config.db_dir,
    #    prefix="soc_parquet",
    #    force_refresh=True,
    #)
    uvicorn.run("classifAI_API.fast_api:app", port=8000, log_level="info")
