"""Functions to initiate the API endpoints."""

import csv
from csv import DictReader
from io import StringIO
from typing import Annotated

from cachetools import TTLCache, cached
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse

from classifai.api import API
from classifai.embedding import EmbeddingHandler

cache = TTLCache(
    maxsize=100, ttl=60
)  # Cache with a maximum size of 100 items and a TTL of 60 seconds

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


@cached(cache)
async def process_input_csv(file: UploadFile) -> DictReader:
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


async def combine_all_input(data: DictReader) -> list[dict]:
    """Collect every line of dictionary.

    Paramaters
    ----------
    data : DictReader[str]
        Dictionary representation of each csv line.

    Returns
    -------
    lines : list of dictionaries.
    """

    # count = 0
    lines = []
    for line in data:
        # count += 1
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

    input = await process_input_csv(file)
    input = await combine_all_input(input)

    embed = EmbeddingHandler(k_matches=3)

    embed.embed_index(file_name="data/soc-index/soc_title_condensed.txt")

    result = embed.search_index(
        input_data=input,
        embedded_fields=["job_title", "company"],
    )

    processed_result = API.simplify_output(
        output_data=result, input_data=input, id_field="id"
    )

    return processed_result


@app.get("/", description="UI accessibility")
async def docs():
    """Access default page: docs UI."""

    start_page = RedirectResponse(url="/docs")
    return start_page
