"""Functions to initiate the API endpoints."""

import os
from typing import Annotated

from dotenv import dotenv_values
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from google.cloud import storage
from pydantic import BaseModel, Field

from src.classifai.embedding import EmbeddingHandler
from src.classifai.utils import (
    get_secret,
    process_embedding_search_result,
    pull_vdb_to_local,
)

api_type = os.getenv("API_TYPE", default="live")

if api_type == "live":
    DB_DIR = "/tmp/"
    BUCKET_NAME = get_secret(
        "APP_DATA_BUCKET", project_id=os.getenv("PROJECT_ID")
    )
else:
    DB_DIR = "data/db/"
    config = dotenv_values(".env")
    BUCKET_NAME = config["BUCKET_NAME"]


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
    local_dir=DB_DIR,
    prefix="sic_knowledge_base_db/",
    bucket_name=BUCKET_NAME,
)
pull_vdb_to_local(
    client=storage.Client(),
    local_dir=DB_DIR,
    prefix="soc_knowledge_base_db_OLD/",
    bucket_name=BUCKET_NAME,
)


### pydantic model classes that work with FastAPI to type check the input to the API
# Model for a single SOC/SIC row entry
class ClassifaiEntry(BaseModel):
    """Model for a single row of data (SOC or SIC row etc), includes 'id' and 'description' which are expected as str type."""

    id: str = Field(examples=["1"])
    description: str = Field(
        description="User string describing occupation or industry",
        examples=["A butcher's shop"],
    )


# Model for a collection of SIC/SOC entries
class ClassifaiData(BaseModel):
    """Pydantic object which contains list of many SOC/SIC Classifai Entry pydantic models."""

    entries: list[ClassifaiEntry] = Field(
        description="array of SOC/SIC Entries to be classified"
    )


### pydantic model classes that work with FastAPI to type check the output of the API - same for SIC and SOC
class ResultEntry(BaseModel):
    """model for single vdb entry."""

    label: str
    bridge: str
    description: str
    distance: float
    rank: int


class ResultsList(BaseModel):
    """model for ranked list of VDB entries for a single row input."""

    input_id: str
    response: list[ResultEntry]


class ResultsResponseBody(BaseModel):
    """model for set of ranked lists, for all row entries submmitted."""

    data: list[ResultsList]


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
        db_dir=os.path.join(DB_DIR, "soc_knowledge_base_db_OLD"),
        k_matches=n_results,
    )

    query_result = handler.collection.query(
        query_texts=input_desc,
        n_results=handler.k_matches,
    )

    query_result["input_ids"] = input_ids

    processed_result = process_embedding_search_result(
        query_result=query_result, include_bridge=False
    )

    return processed_result


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
        db_dir=os.path.join(DB_DIR, "sic_knowledge_base_db"),
        k_matches=n_results,
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
