"""
User interface app for Surveys team.

Run from root directory terminal with:
python -m flask --app flask_ui/app.py run
"""

import sys

sys.path.append("src/")
import logging

from dotenv import find_dotenv, load_dotenv
from flask import (
    Flask,
    make_response,
    render_template,
    request,
    send_from_directory,
)

from classifai.config import Config
from flask_ui.api import (
    api_call_no_auth,
    api_call_with_auth,
    obtain_oidc_token,
)
from flask_ui.db import db, sqlite_app_config
from flask_ui.lib import create_app, remove_asterisk_labels

load_dotenv(find_dotenv())
config = Config("UI")
config.setup_logging()

logging.info(f"API URL: {config.api_url}")

if not config.validate():
    logging.error("Invalid configuration. Exiting.")
    import sys

    sys.exit(1)


"""Creating app, initialized with config & database."""
app = create_app(
    app=Flask(__name__),
    app_config=config,
    db=db,
    db_config_uri=sqlite_app_config,
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
                headers=obtain_oidc_token(config.oauth_client_id),
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
