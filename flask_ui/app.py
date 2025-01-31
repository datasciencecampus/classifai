"""
User interface app for Surveys team.

Run from root directory terminal with:
python -m flask --app flask_ui/app.py run
"""

import json
import logging
import os

import google.cloud.logging
import requests
from dotenv import dotenv_values
from flask import (
    Flask,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
)
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from classifai.utils import get_secret

env_type = os.getenv("ENV_TYPE", default="dev")
api_type = os.getenv("API_TYPE", default="live")

print(f"Environment type: {env_type}")
print(f"API type: {api_type}")

if env_type == "local":
    config = dotenv_values(".env")
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    OAUTH_CLIENT_ID = config.get("OAUTH_CLIENT_ID")
    API_URL = (
        "http://127.0.0.1:8000"
        if api_type == "local"
        else config.get("API_URL")
    )
    logging.info(f"API URL: {API_URL}")

elif env_type == "dev":
    logger = google.cloud.logging.Client()
    logger.setup_logging()
    assert api_type == "live", "Live frontend doesn't work with local backend"
    API_URL = os.getenv("API_URL")
    logging.info(f"API URL: {API_URL}")
    PROJECT_ID = os.getenv("PROJECT_ID")
    OAUTH_CLIENT_ID = get_secret("app_oauth_client_id", project_id=PROJECT_ID)


def _obtain_oidc_token(oauth_client_id):
    """Obtain OIDC authentication token."""

    open_id_connect_token = id_token.fetch_id_token(Request(), oauth_client_id)
    headers = {"Authorization": "Bearer {}".format(open_id_connect_token)}
    return headers


def api_call_no_auth(data: dict, url: str) -> str:
    """Return data from local fastapi.

    Parameters
    ----------
    data: dict
        User-input json extracted from csv file sent from front end to flask
    url: str
        The URL of the API endpoint
    """
    logging.info("Getting the results from the API")

    response = requests.request(url=url, method="POST", json=data)
    json_string = str(response.text)

    try:
        # Parse the JSON string into a Python dictionary
        input_json = json.loads(json_string)

        # Return the JSON
        return input_json

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400


def api_call_with_auth(data: dict, url: str, headers: dict) -> str:
    """Return data from live fastapi.

    Parameters
    ----------
    data: dict
        User-input json extracted from csv file sent from front end to flask
    url: str
        The URL of the API endpoint
    headers: dict
        passes the auth details for the live server
    """
    logging.info("Getting the results from the API")

    response = requests.request(
        url=url, method="POST", json=data, headers=headers
    )

    json_string = str(response.text)

    try:
        # Parse the JSON string into a Python dictionary
        input_json = json.loads(json_string)

        # Return the JSON
        return input_json

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400


app = Flask(__name__)


@app.route("/")
def index():
    """Single page application.

    Returns
    -------
        html: Page
    """
    logging.info("Serving the application")
    response = make_response(render_template("_index.html"))
    response.headers.set(
        "Cache-control", "must-revalidate, max-age=86400"
    )  # 24 hours
    return response


@app.route("/<path:path>")
def serve_file(path):
    """Generate template.

    Args:
        path (str): A path

    Returns
    -------
        html: Page
    """
    return send_from_directory(".", path)


@app.route("/predict_sic", methods=["POST"])
def predict_sic():
    """Retrieve the JSON of SIC code results.

    Returns
    -------
        json: SIC results
    """
    logging.info("Getting SIC codes")

    # getting the users uploaded ['id, 'industry_description']
    input_json = request.json
    input_json = [
        {key: d[key] for key in ["id", "industry_description"]}
        for d in input_json
    ]  # extracting the necessary keys
    json_request_body = {
        "entries": input_json
    }  # correctly formatted for fastapi request

    # if in 'dev' env_type call real api with auth, if in 'local' env_type call localhost api with no auth
    if api_type == "live":
        logging.info("Calling LIVE fastapi server")
        return api_call_with_auth(
            json_request_body,
            f"{API_URL}/sic",
            headers=_obtain_oidc_token(OAUTH_CLIENT_ID),
        )

    elif api_type == "local":
        logging.info("Calling LOCAL fastapi server")
        # return send_from_directory("static", "mock_sic_response.json")
        return api_call_no_auth(
            json_request_body,
            f"{API_URL}/sic",
        )

    else:  # api_type == 'mock'
        logging.info("Returning mock api data")
        return send_from_directory("static", "mock_sic_response.json")
