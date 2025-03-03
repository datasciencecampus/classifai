"""Utility to classify chunks of input data."""

import json
import math
from ast import literal_eval

import numpy as np
import pandas as pd
import requests
from google.auth.transport.requests import Request
from google.cloud import secretmanager
from google.oauth2 import id_token

project_id = "ons-dsc-classifai-prod"
version_id = "latest"


def access_secret_version(project_id, secret_id, version_id):
    """Access required secret version from GCP Secrets Manager."""

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")

    return payload


def _obtain_oidc_token():
    """Obtain OIDC authentication token."""

    secret_id = "app_oauth_client_id"
    oauth_client_id = access_secret_version(project_id, secret_id, version_id)
    open_id_connect_token = id_token.fetch_id_token(Request(), oauth_client_id)
    headers = {"Authorization": "Bearer {}".format(open_id_connect_token)}

    return headers


def _load_data():
    """Load ingest data to be processed."""

    secret_id = "INGEST_DATA_BUCKET"
    ingest_data_bucket = access_secret_version(
        project_id, secret_id, version_id
    )
    data = pd.read_csv(f"{ingest_data_bucket}/coding_df.csv")
    data = data[["bus_id", "desc"]]
    data.columns = ["id", "industry_description"]

    return data


def classify_data():
    """Classify input data in batches."""

    data = _load_data()
    length = len(data)
    batch_size = 25
    num_batches = math.ceil(length / batch_size)
    combined_response = {"data": []}

    for batch_num, (_, chunk) in enumerate(
        data.groupby(np.arange(len(data)) // batch_size)
    ):
        print(f"Processing chunk {batch_num+1} of {num_batches}")
        chunk_filepath = f"data/chunk_{batch_size}.csv"
        chunk.to_csv(chunk_filepath)

        headers = _obtain_oidc_token()

        params = {
            "n_results": "2",
        }

        files = {
            "file": (chunk_filepath, open(chunk_filepath, "rb"), "text/csv"),
        }

        secret_id = "APP_URL"
        url = access_secret_version(project_id, secret_id, version_id)
        response = requests.post(
            f"{url}/sic", params=params, headers=headers, files=files
        )
        response = literal_eval(response.text)
        combined_data = combined_response.get("data", []) + response.get(
            "data", []
        )
        combined_response = {"data": combined_data}

    with open("data/labelled_full.json", "w") as f:
        json.dump(combined_response, f)

    return combined_response


def process_response(response: dict):
    """Process combined classifier responses to csv."""

    labelled_list = []

    for el in response["data"]:
        labelled_list.append(
            [
                el["input_id"],
                el["response"][0]["label"],
                el["response"][0]["description"],
                el["response"][0]["distance"],
                el["response"][1]["label"],
                el["response"][1]["description"],
                el["response"][1]["distance"],
            ]
        )

    columns = [
        "id",
        "label_1",
        "desc_1",
        "distance_1",
        "label_2",
        "desc_2",
        "distance_2",
    ]
    df = pd.DataFrame(data=labelled_list, columns=columns)

    inputs = _load_data()
    output = pd.merge(inputs, df, on="id")

    secret_id = "OUTPUTS_DATA_BUCKET"
    outputs_data_bucket = access_secret_version(
        project_id, secret_id, version_id
    )

    output.to_csv(f"{outputs_data_bucket}/coding_df_APP_CODED.csv")


if __name__ == "__main__":
    response = classify_data()
    process_response(response)
