"""ClassifAI utility classes and functions."""

import os
import shutil
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
from google.cloud import storage
from google.cloud.secretmanager import SecretManagerServiceClient


class DB_Updater:
    """Class of methods to maintain updated DB files remotely."""

    def __init__(
        self,
        storage_client: storage.Client = storage.Client(),
        local_filepath: str = "/tmp",
        bucket_folder: str = "db/",
    ):
        """Initialise DB_Updater class.

        Parameters
        ----------
        storage_client : storage.Client
            GCS Client object.
        local_filepath : str
            Filepath to project DB directory.
        bucket_name : str
            Name of designated GCS bucket.
        bucket_folder : str
            Bucket folder to be used/created for storage.
        """

        self.storage_client = storage_client
        self.local_filepath = local_filepath
        self.bucket_folder = bucket_folder

        self.bucket_name = get_secret(
            "APP_DATA_BUCKET", project_id=os.getenv("PROJECT_ID")
        )

    def delete_existing_gcs_bucket_folder(self):
        """Delete previous db and files.

        Notes
        -----
        This method does not fail if there is no existing matching folder.
        """

        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.bucket_folder)
        for b in blobs:
            b.delete()

        print("Existing remote db files removed.")

    def list_all_db_files(self) -> list[str]:
        """Iterate over db folder to list all files in 2 layers.

        Returns
        -------
        all_filepaths_to_copy : list[str]
            List of all filepaths in folder root and subsequent
            folder layer (DB cache files).
        """

        all_filepaths_to_copy = []
        for layer1_el in os.listdir(self.local_filepath):
            if "." in layer1_el:
                all_filepaths_to_copy.append(
                    f"{self.local_filepath}/{layer1_el}"
                )
            else:
                for layer2_el in os.listdir(
                    f"{self.local_filepath}/{layer1_el}"
                ):
                    all_filepaths_to_copy.append(
                        f"{self.local_filepath}/{layer1_el}/{layer2_el}"
                    )

        print("Local DB files collected.")

        return all_filepaths_to_copy

    def write_local_files_to_gcs_bucket(
        self, all_filepaths_to_copy: list[str]
    ):
        """Write contents at list of local filepaths to GCS bucket.

        Parameters
        ----------
        all_filepaths_to_copy : list[str]
            List of all filepaths in folder root and subsequent
            folder layer (DB cache files).
        """

        bucket = self.storage_client.bucket(self.bucket_name)

        for entry in all_filepaths_to_copy:
            blob = bucket.blob(
                self.bucket_folder + entry.split(f"{self.local_filepath}/")[1]
            )
            blob.upload_from_filename(entry)

        print("DB files successfully written to GCS bucket.")

    def update(self):
        """Update remote db from local changes."""
        self.delete_existing_gcs_bucket_folder()
        local_db_files = self.list_all_db_files()
        self.write_local_files_to_gcs_bucket(local_db_files)


def get_secret(secret_name: str, project_id: str = "classifai-sandbox"):
    """Access GCP Secret Manager secret value.

    Parameters
    ----------
    secret_name : string
        Name of secret to access.
    project_id : string
        GCP project name.

    Returns
    -------
    secrets : str|json
        Secret value.
    """

    env_var_project_id = os.getenv("PROJECT_ID")
    if env_var_project_id:
        project_id = env_var_project_id
    path = f"projects/{project_id}/secrets"
    name = "/".join((path, secret_name, "versions", "latest"))
    client = SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": name})
    secret_value = response.payload.data.decode("UTF-8")

    return secret_value


def pull_vdb_to_local(
    client: storage.Client,
    prefix: str = "db/",
    local_dir: str = "/tmp/",
    vdb_file: str = "chroma.sqlite3",
):
    """Pull only sqlite3 database / vector store to local /tmp dir.

    Parameters
    ----------
    client : storage.Client
        GCS client object
    prefix : str
        GCS bucket folder
        Default: 'db/'
    local_dir : str
        Location of local/instance temporary directory
        Default: '/tmp/'
    vdb_file : str
        Name and extension of vector database
        Default: 'chroma.sqlite3'
    """

    # refresh local folder destination
    try:
        shutil.rmtree(local_dir + prefix)
        print("Deleted previous collection cache.")
    except OSError:
        pass

    os.mkdir(local_dir + prefix)

    bucket_name = get_secret(
        "APP_DATA_BUCKET", project_id=os.getenv("PROJECT_ID")
    )
    bucket = client.bucket(bucket_name=bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if filename == vdb_file:
            # Download to local
            blob.download_to_filename(local_dir + prefix + filename)


def process_embedding_search_result(
    query_result: chromadb.QueryResult,
) -> dict:
    """Structure embedding search result into JSON format.

    Parameters
    ----------
    query_result : chromadb.QueryResult
        ChromaDB vector search result format

    Returns
    -------
    processed_result : dict
        Dictionary format of Chroma DB vector search
    """
    processed_result = {
        "data": [
            {
                "input_id": input_id,
                "response": [
                    {
                        "label": query_result["metadatas"][i][j]["label"],
                        "bridge": query_result["metadatas"][i][j]["bridge"],
                        "description": description,
                        "distance": query_result["distances"][i][j],
                        "rank": j + 1,
                    }
                    for j, description in enumerate(
                        query_result["documents"][i]
                    )
                ],
            }
            for i, input_id in enumerate(query_result["input_ids"])
        ]
    }

    return processed_result


def setup_vector_store(classification: str, distance_metric: str = "l2"):
    """Create new vector store collection in cloud bucket.

    Parameters
    ----------
    classification : string
        Classification task: (required) Current options being 'sic_5_digit',
        'sic_5_digit_extended', or 'soc'

    Raises
    ------
    TypeError
        Raised if unrecognised classification task is entered.
    """

    google_api_key = get_secret(
        "GOOGLE_API_KEY", project_id=os.getenv("PROJECT_ID")
    )
    app_data_bucket = get_secret(
        "APP_DATA_BUCKET", project_id=os.getenv("PROJECT_ID")
    )

    if classification == "sic_5_digit":
        input = pd.read_csv(f"gs://{app_data_bucket}/sic_5_digit.csv")

    elif classification == "sic_5_digit_extended":
        input = pd.read_csv(f"gs://{app_data_bucket}/sic_5_digit_extended.csv")

    elif classification == "sic_knowledge_base":
        input = pd.read_csv(f"gs://{app_data_bucket}/sic_knowledge_base.csv")

    elif classification == "soc":
        input = pd.read_csv(
            f"gs://{app_data_bucket}/soc_title_condensed.txt",
            sep=": ",
            header=None,
        )
        input.columns = ["soc_code", "description"]

    else:
        raise TypeError(
            "You must declare a classification type: currently 'sic_5_digit', 'sic_5_digit_extended', 'sic_knowledge_base' or 'soc' are acceptable values."
        )
    input.reset_index(inplace=True)

    input.rename(
        columns={
            "index": "id",
            "sic_code": "label",
            "soc_code": "label",
        },
        inplace=True,
    )

    task_type = "CLASSIFICATION"

    documents = input["description"].to_list()
    ids = input["id"].astype(str).to_list()

    metadata = [
        {"hnsw": distance_metric, "label": label, "bridge": bridge}
        for label, bridge in zip(input["label"], input["bridge"])
    ]

    # clean out local tmp sub-folder
    db_loc = "/tmp/db"
    db_loc_path = Path(db_loc)
    for folder in db_loc_path.glob("*"):
        if folder.is_dir():
            shutil.rmtree(folder)
            print("Deleted previous collection cache.")

    # create client
    chroma_client = chromadb.PersistentClient(path=db_loc)
    collection_name = "classifai-collection"

    try:
        chroma_client.delete_collection(collection_name)
        print("Existing ChromaDB collection deleted.")
    except ValueError:
        pass

    model_name = "models/text-embedding-004"
    embedding_function = (
        embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=google_api_key, model_name=model_name, task_type=task_type
        )
    )

    print("Creating new collection locally")
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    collection.add(ids=ids, documents=documents, metadatas=metadata)

    updater = DB_Updater(
        local_filepath=db_loc, bucket_folder=f"{classification}_db/"
    )
    updater.update()
