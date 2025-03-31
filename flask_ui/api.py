"""
API module.

Contains the functions & logic which is used by our flask app in requesting from the FastAPI endpoint.
"""

import json
import logging

import requests
from flask import jsonify
from google.auth.transport.requests import Request
from google.oauth2 import id_token


def obtain_oidc_token(oauth_client_id):
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
