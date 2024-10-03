"""
User interface app for Surveys team.

Run from root directory terminal with:
python -m flask --app flask_ui/app.py run
"""

import io
import json
import os

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from gcp_iap_auth.user import User, UserIAPClient

load_dotenv()

OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")

API_URL = os.getenv("APP_URL")

CREDENTIAL_PATH = os.getenv("CREDENTIAL_PATH")

app = Flask(__name__)


def get_json_results_from_api(file: str) -> str:
    """Process data on live API and return response as string.

    Parameters
    ----------
    file : UploadFile
        User-input csv data.
    """
    user = User(
        oauth_client_id=OAUTH_CLIENT_ID,
        oauth_client_secret=OAUTH_CLIENT_SECRET,
        credentials_path=CREDENTIAL_PATH,
    )

    iap_client = UserIAPClient(user=user)

    files = {"file": file}

    response = iap_client.request(
        url=API_URL,
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


@app.route("/predict_soc", methods=["POST"])
def predict_soc():
    """Retrieve the JSON of SOC code results (mocked).

    Returns
    -------
        json: mocked SOC results
    """
    jobs = request.json
    # print(jobs)
    jobs_df = pd.DataFrame(jobs)
    jobs_df = jobs_df.rename(
        columns=dict(title="job_title", employer="company")
    )
    jobs_df = jobs_df.drop(columns=["wage", "description", "supervision"])
    csv_buffer = io.StringIO()
    jobs_df.to_csv(csv_buffer, index=False)
    jobs_csv = csv_buffer.getvalue()
    # print(jobs_csv)

    response = get_json_results_from_api(jobs_csv)
    return response
