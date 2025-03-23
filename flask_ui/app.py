"""
User interface app for Surveys team.

Run from root directory terminal with:
python -m flask --app flask_ui/app.py run
"""

import json
import logging

import requests
from dotenv import find_dotenv, load_dotenv
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

from src.classifai.config import Config

load_dotenv(find_dotenv())
config = Config("UI")
config.setup_logging()

logging.info(f"API URL: {config.api_url}")

if not config.validate():
    logging.error("Invalid configuration. Exiting.")
    import sys

    sys.exit(1)


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


def remove_asterisk_labels(data):
    """Remove any ranked soc codes from the ranking.

    Parameters
    ----------
    data: dict
        the json object / python dict returned by a succeful call to FastAPI SIC or SOC endpoint
    """

    # for each rows returned ranking, iterate through the ranking and remove ranked items where the label value is *
    for entry in data["data"]:
        entry["response"] = [
            response
            for response in entry["response"]
            if response["label"] != "*"
        ]

    # do the same for the deduplicated data
    for entry in data["deduplicated_data"]:
        entry["response"] = [
            response
            for response in entry["response"]
            if response["label"] != "*"
        ]

    return data


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


# endpoint for predicting SOC codes
@app.route("/predict_soc", methods=["POST"])
def predict_soc():
    """Retrieve the JSON of SOC code results.

    Returns
    -------
        json: SOC results
    """
    logging.info("Getting SOC codes")

    # getting the users uploaded ['id, 'description']
    input_json = request.json
    input_json = [
        {key: d[key] for key in ["id", "description"]} for d in input_json
    ]  # extracting the necessary keys
    json_request_body = {
        "entries": input_json
    }  # correctly formatted for fastapi request

    # if in 'live' api_type call real api with auth, if in 'local' api_type call localhost api with no auth
    if config.api_type == "live":
        logging.info("Calling LIVE fastapi server")

        # 324-QUICKFIX-removing asterisk labelled entries from ranking
        return remove_asterisk_labels(
            api_call_with_auth(
                json_request_body,
                f"{config.api_url}/soc",
                headers=_obtain_oidc_token(config.oauth_client_id),
            )
        )

    elif config.api_type == "local":
        logging.info("Calling LOCAL fastapi server")

        # 324-QUICKFIX-removing asterisk labelled entries from ranking
        return remove_asterisk_labels(
            api_call_no_auth(
                json_request_body,
                f"{config.api_url}/soc",
            )
        )

    else:  # api_type == 'mock'
        logging.info("Returning mock api data")
        return send_from_directory("static", "mock_soc_response.json")


# endpoint for predicting SIC codes
@app.route("/predict_sic", methods=["POST"])
def predict_sic():
    """Retrieve the JSON of SIC code results.

    Returns
    -------
        json: SIC results
    """
    logging.info("Getting SIC codes")

    # getting the users uploaded ['id, 'description']
    input_json = request.json
    input_json = [
        {key: d[key] for key in ["id", "description"]} for d in input_json
    ]  # extracting the necessary keys
    json_request_body = {
        "entries": input_json
    }  # correctly formatted for fastapi request

    # if in 'live' api_type call real api with auth, if in 'local' api_type call localhost api with no auth
    if config.api_type == "live":
        logging.info("Calling LIVE fastapi server")
        return api_call_with_auth(
            json_request_body,
            f"{config.api_url}/sic",
            headers=_obtain_oidc_token(config.oauth_client_id),
        )

    elif config.api_type == "local":
        logging.info("Calling LOCAL fastapi server")
        # return send_from_directory("static", "mock_sic_response.json")
        return api_call_no_auth(
            json_request_body,
            f"{config.api_url}/sic",
        )

    else:  # api_type == 'mock'
        logging.info("Returning mock api data")
        return send_from_directory("static", "mock_sic_response.json")
