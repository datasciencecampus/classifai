"""
Utility script to classify occupation/job descriptions programmatically using the back end FastAPI ClassifAI servers (without using frontend UI).

Required Inputs:
- PROJECT_ID: Google Cloud Project ID
- VERSION_ID: Version ID for secrets (default "latest")
- FASTAPI_URL: URL for the FastAPI endpoint script should contact to request classifications
- LOCAL_STORAGE: Boolean flag for using local storage and not cloud storage
- LOCAL_STORAGE_FILEPATH: IF LOCAL_STORAGE is set to TRUE, input file will be loaded from this filepath
- BUCKET_SECRET: Name of the bucket secret to get access cloud bucket when accessing input file from cloud
- BUCKET_LOAD_PATH: Path to the file to load from bucket
- BUCKET_SAVE_FOLDER: Bucket location to save data to (if LOCAL_STORAGE is set to FALSE)
- BUCKET_SAVE_FILE: Base name for saved files
- BATCH_SIZE: Number of entries to process in each batch
- SAVE_PER_BATCH: Boolean flag to save results after each batch
- N_RESULTS: Number of ranked search results to request from the API, per job description


Format of main input data:
- The file of occupation/job descriptions should be of .json format.
- This is the file pointed to by LOCAL_STORAGE_PATH (or BUCKET_LOAD_PATH when loading from cloud bucket)
- It should follow the json structure of (excluding '---'):

---
    {
    "entries": [
        {
        "id": "1",
        "description": "butcher"
        },
        {
        "id": "2",
        "description": "baker"
        },
        ...
        {
        "id": "9999",
        "description": "candlestick maker"
        }
    ]
    }
---


Files Created:
1. Batch Files (when SAVE_PER_BATCH=True):
   - Local: ./{BUCKET_SAVE_FILE}_batch_{batch_number}.json for each processed batch
   - GCS Bucket (when LOCAL_STORAGE=False): {BUCKET_SAVE_FOLDER}/{BUCKET_SAVE_FILE}_batch_{batch_number}.json

2. Combined Results File:
   - Local: ./{BUCKET_SAVE_FILE}_full.json containing all processed data
   - GCS Bucket (when LOCAL_STORAGE=False): {BUCKET_SAVE_FOLDER}/{BUCKET_SAVE_FILE}_full.json

3. Temporary Local File (when LOCAL_STORAGE=False):
   - {LOCAL_STORAGE_FILEPATH} - created when downloading data from GCS bucket
"""

import json
import time
from ast import literal_eval

import requests
from google.auth.transport.requests import Request
from google.cloud import secretmanager, storage
from google.oauth2 import id_token

#####
###ENVIRONMENT VARIABLES TO BE SET BEFORE RUNNING
PROJECT_ID = ""
VERSION_ID = "latest"
FASTAPI_URL = ""

LOCAL_STORAGE = False
LOCAL_STORAGE_FILEPATH = ""  # name of the file to load locally and save locally when downloading from cloud

BUCKET_SECRET = ""  # name of the bucket secret to get
BUCKET_LOAD_PATH = ""  # name of the file to load from bucket

BUCKET_SAVE_FOLDER = ""  # bucket location to save data to
BUCKET_SAVE_FILE = (
    ""  # code adds .json and automatically saves full file or batch files etc
)

BATCH_SIZE = 100
SAVE_PER_BATCH = True
N_RESULTS = 20


# AUTH FUNCTIONS
################
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
    oauth_client_id = access_secret_version(PROJECT_ID, secret_id, VERSION_ID)
    open_id_connect_token = id_token.fetch_id_token(Request(), oauth_client_id)
    headers = {"Authorization": "Bearer {}".format(open_id_connect_token)}

    return headers


# DATA LOADING FUNCTION
#######################


def download_from_gcs(
    project_id, bucket_name, source_blob_name, destination_file_name
):
    """Download a blob from the bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to {destination_file_name}"
    )


def upload_to_gcs(
    project_id, bucket_name, file_to_upload, destination_blob_name
):
    """Upoad a file to the bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_to_upload)
    print(f"Uploaded file {destination_blob_name} to bucket {bucket_name}")


def _load_data_bucket():
    """Load data from the online gcloud bucket storage."""

    ingest_data_bucket = access_secret_version(
        PROJECT_ID, BUCKET_SECRET, VERSION_ID
    )

    download_from_gcs(
        PROJECT_ID,
        ingest_data_bucket,
        BUCKET_LOAD_PATH,
        LOCAL_STORAGE_FILEPATH,
    )

    with open(LOCAL_STORAGE_FILEPATH) as file:
        data = json.load(file)
        entries = data["entries"][0:200]
        for ii in range(0, len(entries), BATCH_SIZE):
            yield entries[ii : ii + BATCH_SIZE]


def _load_data_local():
    """Load data from the local file storage."""

    with open(LOCAL_STORAGE_FILEPATH, "r") as file:
        data = json.load(file)
        entries = data["entries"]
        for ii in range(0, len(entries), BATCH_SIZE):
            yield entries[ii : ii + BATCH_SIZE]


# DATA PROCESSING FUNCTIONS
#######################
def classify_data():
    """Classify input data in batches."""

    # create the genertor functions
    if LOCAL_STORAGE:
        print(f"looking for local file {LOCAL_STORAGE_FILEPATH}")
        data = _load_data_local()
    else:
        print(f"accessing cloud bucket {BUCKET_LOAD_PATH}")
        data = _load_data_bucket()

    combined_response = {"data": []}

    for batch_num, batch in enumerate(data):
        # obtain oidc token again because it expires after 1 use (or its has a fixed lifespan if not 1 time use?)
        headers = _obtain_oidc_token()

        params = {
            "n_results": N_RESULTS,
        }

        json_data = {"entries": batch}

        url = FASTAPI_URL

        # take 5 atttempts at calling the API
        escape_batch = False
        for attempt in range(5):
            response = requests.post(
                f"{url}/soc", params=params, headers=headers, json=json_data
            )
            if response.status_code == 200:
                print("got 200 response and data from api")
                break
            elif attempt + 1 == 5:
                print(
                    f"Attempted request no.{attempt+1} for batch {batch_num+1} unsuccesful with code {response.status_code}"
                )
                print("5 unsuccesful attempts on batch, skipping batch")
                escape_batch = True
            else:
                print(
                    f"Attempted request no.{attempt+1} for batch {batch_num+1} unsuccesful with code {response.status_code}"
                )
                print(
                    "sleeping for 5s for api usage cooldown before re-requesting"
                )
                time.sleep(5)

        if escape_batch:
            continue

        response = literal_eval(response.text)

        combined_data = combined_response.get("data", []) + response.get(
            "data", []
        )
        combined_response = {"data": combined_data}

        print(
            f"completed batch {batch_num+1}, processed {(batch_num+1)*BATCH_SIZE} entries"
        )

        if SAVE_PER_BATCH:
            with open(
                f"./{BUCKET_SAVE_FILE}_batch_{batch_num+1}.json", "w"
            ) as f:
                json.dump(response.get("data", []), f)

            if not LOCAL_STORAGE:
                ingest_data_bucket = access_secret_version(
                    PROJECT_ID, BUCKET_SECRET, VERSION_ID
                )
                upload_to_gcs(
                    PROJECT_ID,
                    ingest_data_bucket,
                    f"./{BUCKET_SAVE_FILE}_batch_{batch_num+1}.json",
                    f"{BUCKET_SAVE_FOLDER}/{BUCKET_SAVE_FILE}_batch_{batch_num+1}.json",
                )

    # save the whole combined responses to file
    with open(f"./{BUCKET_SAVE_FILE}_full.json", "w") as f:
        json.dump(combined_response, f)

    # if we're using cloud buckets push to the cloud
    if not LOCAL_STORAGE:
        ingest_data_bucket = access_secret_version(
            PROJECT_ID, BUCKET_SECRET, VERSION_ID
        )
        upload_to_gcs(
            PROJECT_ID,
            ingest_data_bucket,
            f"./{BUCKET_SAVE_FILE}_full.json",
            f"{BUCKET_SAVE_FOLDER}/{BUCKET_SAVE_FILE}_full.json",
        )

    return 0


if __name__ == "__main__":
    response = classify_data()
