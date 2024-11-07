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
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
)
from google.cloud import secretmanager

env_type = os.getenv("ENV_TYPE", default="local-noauth")

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


if env_type == "dev":
    logger = google.cloud.logging.Client()
    logger.setup_logging()

    OAUTH_CLIENT_ID = "NA"
    OAUTH_CLIENT_SECRET = "NA"  # pragma: allowlist secret
    API_URL = "https://classifai-sandbox.nw.r.appspot.com"
    CREDENTIAL_PATH = "./credentials.json"
    logging.info("Loaded global variables in dev")
if env_type == "dev-noauth":
    logger = google.cloud.logging.Client()
    logger.setup_logging()

    API_URL = "https://classifai-sandbox.nw.r.appspot.com"
    CREDENTIAL_PATH = "./credentials.json"
    logging.info("Loaded global variables in dev-noauth")
elif env_type == "local":
    from gcp_iap_auth.user import User, UserIAPClient

    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    load_dotenv()
    OAUTH_CLIENT_ID = access_secret_version(
        "projects/14177695902/secrets/FLASK_CLIENT_ID/versions/1"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    OAUTH_CLIENT_SECRET = access_secret_version(
        "projects/14177695902/secrets/FLASK_CLIENT_SECRET/versions/1"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    API_URL = "https://classifai-sandbox.nw.r.appspot.com"
    CREDENTIAL_PATH = "./credentials.json"
    logging.info("Loaded global variables in local")
elif env_type == "local-noauth":
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    load_dotenv()

    API_URL = "https://classifai-sandbox.nw.r.appspot.com"
    CREDENTIAL_PATH = "./credentials.json"
    logging.info("Loaded global variables in local-noauth")


app = Flask(__name__)


def api_call_with_auth(file: str, url: str) -> str:
    """Process data on live API and return response as string.

    Parameters
    ----------
    file : UploadFile
        User-input csv data.
    url: str
        The URL of the API endpoint
    """
    logging.info("Getting the results from the API")

    user = User(
        oauth_client_id=OAUTH_CLIENT_ID,
        oauth_client_secret=OAUTH_CLIENT_SECRET,
        credentials_path=CREDENTIAL_PATH,
    )

    iap_client = UserIAPClient(user=user)

    files = {"file": file}

    # print(files)
    response = iap_client.request(
        url=url,
        method="POST",
        files=files,
    )

    json_string = str(response.text)

    try:
        # Parse the JSON string into a Python dictionary
        input_json = json.loads(json_string)

        # Return the JSON
        return input_json

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400


def api_call_no_auth(file: str, url: str) -> str:
    """Process data on live API and return response as string.

    Parameters
    ----------
    file : UploadFile
        User-input csv data.
    url: str
        The URL of the API endpoint
    """
    logging.info("Getting the results from the API w'out auth")

    files = {"file": file}

    response = requests.request(
        url=url,
        method="POST",
        files=files,
    )

    json_string = str(response.text)

    try:
        # Parse the JSON string into a Python dictionary
        input_json = json.loads(json_string)

        # Return the JSON
        return input_json

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400


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

    if env_type == "local":
        jobs_csv = csv_buffer.getvalue()
        return api_call_with_auth(jobs_csv, f"{API_URL}/sic")
    elif env_type in ["dev-noauth", "local-noauth"]:
        jobs_csv = csv_buffer.getvalue()
        response = api_call_no_auth(jobs_csv, f"{API_URL}/sic")
        return response
    else:
        return send_from_directory("static", "mock_sic_response.json")


@app.route("/predict_soc", methods=["POST"])
def predict_soc():
    """Retrieve the JSON of SOC code results.

    Returns
    -------
        json: SOC results
    """
    logging.info("Getting SOC codes")

    jobs = request.json
    # print(jobs)
    input_df = pd.DataFrame(jobs)
    input_df = input_df.rename(
        columns=dict(title="job_title", employer="company")
    )
    input_df = input_df.drop(columns=["wage", "description", "supervision"])
    csv_buffer = io.StringIO()
    input_df.to_csv(csv_buffer, index=False)
    jobs_csv = csv_buffer.getvalue()

    if env_type == "local":
        response = api_call_with_auth(jobs_csv, f"{API_URL}/soc")
    else:
        mocked_results = {
            "data": [
                {
                    "input_id": id,
                    "response": [
                        {
                            "label": "1234",
                            "description": "worker",
                            "distance": 0.5,
                        }
                    ]
                    * 4,
                }
                for id in input_df["id"]
            ]
        }
        response = jsonify(mocked_results)
    return response


# if __name__ == "__main__":
#     app.run()
