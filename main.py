"""Functions to initiate the API endpoints."""

import csv
from csv import DictReader
from io import StringIO
from typing import Annotated

# from cachetools import TTLCache, cached
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from google.cloud import storage

from src.classifai.embedding import EmbeddingHandler
from src.classifai.utils import (
    process_embedding_search_result,
    pull_vdb_to_local,
)

# cache = TTLCache(
#     maxsize=100, ttl=60
# )  # Cache with a maximum size of 100 items and a TTL of 60 seconds

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


# @cached(cache)
async def _process_input_csv(file: UploadFile) -> DictReader:
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


async def _combine_all_input(data: DictReader) -> list[dict]:
    """Collect every line of dictionary.

    Paramaters
    ----------
    data : DictReader[str]
        Dictionary representation of each csv line.

    Returns
    -------
    lines : list of dictionaries.
    """

    lines = []
    for line in data:
        line = {k: v for k, v in line.items() if k != "id"}
        lines.append(line)

    return lines


@app.post("/soc", description="programmatic endpoint")
async def soc(
    file: Annotated[UploadFile, File(description="User input: csv")],
) -> dict:
    """Label input data using programmatic endpoint.

    Parameters
    ----------
    file : UploadFile
        User-provided csv file.

    Returns
    -------
    processed_result : dict
        Dictionary of top-k closest roles to input jobs.
    """

    input = await _process_input_csv(file)
    input = await _combine_all_input(input)
    input = [f'{x["job_title"]} - {x["company"]}' for x in input]
    input_size = len(input)

    pull_vdb_to_local(client=storage.Client())

    handler = EmbeddingHandler()

    query_result = handler.collection.query(
        query_texts=input,
        n_results=handler.k_matches,
    )

    remove_keys = ["metadatas", "embeddings", "uris", "data", "included"]

    for key in remove_keys:
        del query_result[key]

    query_result["inputs"] = input

    processed_result = process_embedding_search_result(
        query_result, input_size
    )

    return processed_result


@app.get("/", description="UI accessibility")
async def docs():
    """Access default page: docs UI."""

    start_page = RedirectResponse(url="/docs")
    return start_page
