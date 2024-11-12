"""
User interface app for Surveys team.

Run from root directory terminal with:
python -m flask --app flask_ui/app.py run
"""

import io
import json
import logging
import os

import google.cloud.logging
import pandas as pd
import requests
from flask import (
    Flask,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
)
from google.auth.transport.requests import Request
from google.cloud import secretmanager
from google.oauth2 import id_token

env_type = os.getenv("ENV_TYPE", default="dev")
API_URL = os.getenv(
    "API_URL", default="https://classifai-sandbox.nw.r.appspot.com"
)

print(f"Environment type: {env_type}")


def access_secret_version(resource_id):
    """
    Access the payload for the given secret version if one exists.

    The version can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    """
    # Create a client
    client = secretmanager.SecretManagerServiceClient()

    # Initialize request argument(s)
    request = secretmanager.AccessSecretVersionRequest(
        name=resource_id,
    )

    # Make the request
    response = client.access_secret_version(request=request)

    # Handle the response
    return response.payload.data.decode("UTF-8")


def _obtain_oidc_token(oauth_client_id):
    """Obtain OIDC authentication token."""

    open_id_connect_token = id_token.fetch_id_token(Request(), oauth_client_id)
    headers = {"Authorization": "Bearer {}".format(open_id_connect_token)}

    return headers


def api_call_with_auth(file: str, url: str, headers: dict) -> str:
    """Process data on live API and return response as string.

    Parameters
    ----------
    file : UploadFile
        User-input csv data.
    url: str
        The URL of the API endpoint
    """
    logging.info("Getting the results from the API")

    files = {"file": file}

    # print(files)
    response = requests.request(
        url=url, method="POST", files=files, headers=headers
    )

    json_string = str(response.text)

    try:
        # Parse the JSON string into a Python dictionary
        input_json = json.loads(json_string)

        # Return the JSON
        return input_json

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400


if env_type == "local":
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    # this is currently incomplete...
    creds = access_secret_version(
        "projects/14177695902/secrets/auth-credentials/versions/latest"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    with open("./credentials.json", "w") as f:
        f.write(creds)
elif env_type == "dev":
    logger = google.cloud.logging.Client()
    logger.setup_logging()
    OAUTH_CLIENT_ID = access_secret_version(
        "projects/14177695902/secrets/app_oauth_client_id/versions/latest"  # pragma: allowlist secret
    )


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
    input_json = request.json
    input_df = pd.DataFrame(input_json)
    csv_buffer = io.StringIO()
    input_df.to_csv(csv_buffer, index=False)

    if env_type == "dev":
        jobs_csv = csv_buffer.getvalue()
        return api_call_with_auth(
            jobs_csv,
            f"{API_URL}/sic",
            headers=_obtain_oidc_token(OAUTH_CLIENT_ID),
        )
    else:
        return send_from_directory("static", "mock_sic_response.json")


# if __name__ == "__main__":
#     app.run()
