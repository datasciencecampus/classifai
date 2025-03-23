"""Functions to initiate the API endpoints."""

import logging
from pathlib import Path
from typing import Annotated

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from google.cloud import storage

from fast_api.pydantic_models import ClassifaiData, ResultsResponseBody
from src.classifai.config import Config
from src.classifai.embedding import EmbeddingHandler
from src.classifai.search_result_builders import (
    create_deduplicated_response,
    naive_scorer,
    process_embedding_search_result,
)
from src.classifai.utils import pull_vdb_to_local

load_dotenv(find_dotenv())
config = Config("API")
config.setup_logging()

if not config.validate():
    logging.error("Invalid configuration. Exiting.")
    import sys

    sys.exit(1)


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

pull_vdb_to_local(
    client=storage.Client(),
    local_dir=config.db_dir,
    prefix="sic_knowledge_base_db",
    bucket_name=config.bucket_name,
)
pull_vdb_to_local(
    client=storage.Client(),
    local_dir=config.db_dir,
    prefix="soc_knowledge_base_db_OLD",
    bucket_name=config.bucket_name,
)


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
    input_desc = [x.description for x in data.entries]

    handler = EmbeddingHandler(
        vdb_name="classifai-collection",
        db_dir=str(Path(config.db_dir) / "soc_knowledge_base_db_OLD"),
        k_matches=n_results,
        api_key=config.embedding_api_key,
    )

    query_result = handler.collection.query(
        query_texts=input_desc,
        n_results=handler.k_matches,
    )

    query_result["input_ids"] = input_ids

    # format the data response based on the VDB query results
    processed_result = process_embedding_search_result(
        query_result=query_result, include_bridge=False
    )

    # format the deduplicated data response based on the original formatted data
    deduplicated_result = create_deduplicated_response(
        processed_result, naive_scorer
    )

    # return both data and deduplicated data
    return {
        "data": processed_result,
        "deduplicated_data": deduplicated_result,
    }


@app.post("/sic", description="SIC programmatic endpoint")
def sic(
    data: ClassifaiData,
    n_results: Annotated[
        int,
        Query(
            description="The number of results to return per SIC row.",
        ),
    ] = 20,
) -> ResultsResponseBody:
    """Label input data using SIC programmatic endpoint.

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
    input_desc = [x.description for x in data.entries]

    handler = EmbeddingHandler(
        vdb_name="classifai-collection",
        db_dir=str(Path(config.db_dir) / "sic_knowledge_base_db"),
        k_matches=n_results,
        api_key=config.embedding_api_key,
    )

    query_result = handler.collection.query(
        query_texts=input_desc,
        n_results=handler.k_matches,
    )

    query_result["input_ids"] = input_ids

    processed_result = process_embedding_search_result(
        query_result=query_result, include_bridge=True
    )

    return processed_result


@app.get("/", description="UI accessibility")
def docs():
    """Access default page: docs UI."""

    start_page = RedirectResponse(url="/docs")
    return start_page
