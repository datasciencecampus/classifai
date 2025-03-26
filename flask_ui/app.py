"""
User interface app for Surveys team.

Run from root directory terminal with:
python -m flask --app flask_ui/app.py run
"""

import logging

from dotenv import find_dotenv, load_dotenv
from flask import (
    Flask,
    Response,
    make_response,
    render_template,
    request,
    send_from_directory,
)
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from flask_ui.api import api_call_no_auth, api_call_with_auth
from flask_ui.db import db, db_config_uri
from flask_ui.db.lib import get_local_user_credentials
from flask_ui.db.queries import (
    create_job_data,
    create_many_results_many_jobs,
    create_session,
    get_or_create_user,
)
from flask_ui.lib import create_app, remove_asterisk_labels
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


"""Creating app, initialized with config & database."""
app = create_app(
    app=Flask(__name__), app_config=config, db=db, db_config_uri=db_config_uri
)


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


@app.route("/post_session", methods=["POST"])
def post_session():
    """
    POST SESSION VIEW.

    Takes a payload of 'sessionID' & 'jobsData' from the frontend.
    Creates a Session Instance & many Job instances
    """
    logging.info("POST SESSION  CALLED")
    if config.env_type == "local":
        session_id, job_data = request.json
        user_credentials = get_local_user_credentials()
        user = get_or_create_user(db, user_credentials)
        session = create_session(db, user, session_id)
        create_job_data(db, session, job_data)
        logging.info("SESSION & JOBS CREATED SUCCESFULLY")
        return Response(status=200)
    else:
        logging.info(
            f"'{config.env_type}' ENVIRONMENT NOT YET SUPPORTED FOR DB QUERIES"
        )
        return Response(status=501)


@app.route("/post_results", methods=["POST"])
def post_results():
    """
    POST RESULTS VIEW.

    Takes a payload of 'sessionID' & 'resultsData' from the frontend
    """
    logging.info("POST RESULTS CALLED")
    if config.env_type == "local":
        session_id, results_data = request.json
        create_many_results_many_jobs(db, session_id, results_data)
        return Response(status=200)
    else:
        logging.info(
            f"'{config.env_type}' ENVIRONMENT NOT YET SUPPORTED FOR DB QUERIES"
        )
        return Response(status=501)
