"""Functions to define API endpoint behaviour."""

import json

from classifAI import Outputs
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()
tool = Outputs()


@app.get("/", tags=["core_endpoint"])
async def about():
    """Access the API documentation in json format."""
    response = RedirectResponse(url="/openapi.json")

    return response


@app.get("/soc", tags=["task_endpoint"])
async def soc() -> dict:
    """Load SOC output data and filters to required fields.

    Returns
    -------
    data : dict
        Output JSON with required fields only.

    """
    with open("data/soc_mock.json") as f:
        data = json.load(f)
    data = tool.simplify_output(data)

    return data
