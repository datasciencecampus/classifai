"""Functions to initiate the API endpoints."""

import csv
from csv import DictReader
from io import StringIO
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from google.cloud import storage

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


def _process_input_csv(file: UploadFile) -> DictReader:
    """Read csv as strings.

    Parameters
    ----------
    file : UploadFile
        User-provided csv file.

    Returns
    -------
    csvReader : DictReader[str]
        Dictionary representation of each csv line.
    """

    file_content = file.file.read()
    buffer = StringIO(file_content.decode("utf-8"))
    csvReader = csv.DictReader(buffer)

    return csvReader


@app.post("/sic", description="SIC programmatic endpoint")
def sic(
    file: Annotated[UploadFile, File(description="User input: csv")],
    n_results: int = 20,
) -> dict:
    """Label input data using SIC programmatic endpoint.

    Parameters
    ----------
    file : UploadFile
        User-provided csv file.
    n_results: int
        The number of results to return per query

    Returns
    -------
    processed_result : dict
        Dictionary of top n closest roles to input jobs.
    """

    input = list(_process_input_csv(file))
    input_desc = [f'{x["industry_description"]}' for x in input]
    input_ids = [x["id"] for x in input]

    pull_vdb_to_local(
        client=storage.Client(), prefix="sic_5_digit_extended_db/"
    )
    handler = EmbeddingHandler(
        vdb_name="classifai-collection",
        db_dir="/tmp/sic_5_digit_extended_db",
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


@app.post("/soc", description="SOC programmatic endpoint")
def soc(
    file: Annotated[UploadFile, File(description="User input: csv")],
) -> dict:
    """Label input data using SOC programmatic endpoint.

    Parameters
    ----------
    file : UploadFile
        User-provided csv file.

    Returns
    -------
    processed_result : dict
        Dictionary of top-k closest roles to input jobs.
    """

    input = list(_process_input_csv(file))
    input_desc = [f'{x["job_title"]} - {x["company"]}' for x in input]
    input_ids = [x["id"] for x in input]

    pull_vdb_to_local(client=storage.Client(), prefix="soc_db/")
    handler = EmbeddingHandler(
        vdb_name="classifai-collection", db_dir="/tmp/soc_db"
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
