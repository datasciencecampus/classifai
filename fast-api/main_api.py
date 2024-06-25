"""Functions to define API endpoint behaviour."""

import argparse
import warnings

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from classifai import API

app: FastAPI = FastAPI()
tool = API()

# TODO: how best to pass input_filepath as argument to user


@app.get("/", tags=["default_endpoint"])
def about():
    """Access the API documentation in json format."""
    response = RedirectResponse(url="/openapi.json")

    return response


@app.get("/soc", tags=["task_endpoint"])
def soc() -> dict:
    """Load SOC output data and filters to required fields.

    Returns
    -------
    data : dict
        Output JSON with required fields only.

    """
    input_data = tool.jsonify_input()
    embedding_search_result = tool.classify_input(
        input_data,
        embedded_fields=["job_title", "company"],
    )
    output_data = tool.simplify_output(
        embedding_search_result, input_data, id_field="id"
    )

    return output_data


if __name__ == "__main__":
    # TODO: implement user task to direct to req'd endpoint
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", required=False, default="soc", type=str
    )
    args = vars(parser.parse_args())
    endpoint = args["task"]

    compatible_tasks = ["soc"]
    if endpoint in compatible_tasks:
        print(
            f"A dedicated endpoint will be available here: 'http://localhost:8000/{endpoint}'"
        )
    else:
        warnings.warn("Task not recognised. Default endpoint available only.")

    uvicorn.run(app)
