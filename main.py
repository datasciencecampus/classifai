"""Functions to initiate the API endpoints."""

import csv
from io import StringIO
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse

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


@app.post("/soc", description="programmatic endpoint")
def soc(file: Annotated[UploadFile, File(description="User input: csv")]):
    """Accept user input.

    Parameters
    ----------
    file : UploadFile
        User-uploaded csv file.

    Returns
    -------
    output_list_json : list[dict]
        List of dictionaries of (labelled) output data post-processing.
    """
    file_content = file.file.read()
    buffer = StringIO(file_content.decode("utf-8"))
    csvReader = csv.DictReader(buffer)
    output_list_json = []

    # this is where the processing logic will go
    for row in csvReader:
        output_list_json.append(row)

    return output_list_json


@app.get("/", description="UI accessibility")
def docs():
    """Access default page: docs UI."""

    start_page = RedirectResponse(url="/docs")
    return start_page
