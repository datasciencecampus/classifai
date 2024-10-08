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
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

# from gcp_iap_auth.user import User, UserIAPClient

env_type = os.getenv("ENV_TYPE", default="dev")

if env_type == "dev":
    logger = google.cloud.logging.Client()
    logger.setup_logging()
    OAUTH_CLIENT_ID = "NA"
    OAUTH_CLIENT_SECRET = "NA"  # pragma: allowlist secret
    API_URL = os.getenv("API_URL")
    CREDENTIAL_PATH = "NA"
    logging.info("Loaded global variables in dev")
elif env_type == "local":
    from gcp_iap_auth.user import User, UserIAPClient

    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    load_dotenv()
    OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
    OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
    API_URL = os.getenv("API_URL")
    CREDENTIAL_PATH = os.getenv("CREDENTIAL_PATH")
    logging.info("Loaded global variables in local")


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

    print(files)
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


@app.route("/")
def index():
    """Single page application.

    Returns
    -------
        html: Page
    """
    logging.info("Serving the application")
    return render_template("index.html")


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
    # jobs_csv = csv_buffer.getvalue()

    if env_type == "local":
        return send_from_directory("static", "mock_sic_response.json")
        # response = api_call_with_auth(jobs_csv,f"{API_URL}/sic")
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
