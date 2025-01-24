"""Functions to initiate the API endpoints."""

from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from google.cloud import storage
from pydantic import BaseModel, Field

from src.classifai.embedding import EmbeddingHandler
from src.classifai.utils import (
    process_embedding_search_result,
    pull_vdb_to_local,
)

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


### pydantic model classes that work with FastAPI to type check the input to the API


class SICEntry(BaseModel):
    """Model for a single SIC code row, includes 'id' and 'industry_description' which are expected as str type."""

    id: str = Field(examples=["1"])
    industry_description: str = Field(
        description="User string input that describing industry",
        examples=[
            "A butchers business",
        ],
    )


class SICData(BaseModel):
    """Pydantic object which contains list of many SICEntry pydantic models."""

    entries: list[SICEntry] = Field(
        description="array of SIC Entries to be classified"
    )


### pydantic model classes that work with FastAPI to type check the output of the API


class ResultEntry(BaseModel):
    """model for single vdb entry."""

    label: int
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


@app.post("/sic", description="SIC programmatic endpoint")
def sic(
    data: SICData,
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
    data: SICData
        user provided JSON array, contains many SICEntry json objects which are {'id': str, 'industry_description': str}
    n_results: int
        The number of results to return per query

    Returns
    -------
    processed_result : SICResponseBody :: Modelled as a Pydantic response object which provides example json and type checking
        Dictionary (Pydantic Object) of top n closest roles to input jobs.
    """

    input_ids = [x.id for x in data.entries]
    input_desc = [x.industry_description for x in data.entries]

    pull_vdb_to_local(client=storage.Client(), prefix="sic_knowledge_base_db/")
    handler = EmbeddingHandler(
        vdb_name="classifai-collection",
        db_dir="/tmp/sic_knowledge_base_db",
        k_matches=n_results,
    )

    query_result = handler.collection.query(
        query_texts=input_desc,
        n_results=handler.k_matches,
    )

    query_result["input_ids"] = input_ids

    processed_result = process_embedding_search_result(
        query_result=query_result
    )

    return processed_result


@app.get("/", description="UI accessibility")
def docs():
    """Access default page: docs UI."""

    start_page = RedirectResponse(url="/docs")
    return start_page
