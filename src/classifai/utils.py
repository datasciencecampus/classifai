"""ClassifAI utility classes and functions."""

import os

import chromadb
from google.cloud import storage
from google.cloud.secretmanager import SecretManagerServiceClient


class DB_Updater:
    """Class of methods to maintain updated DB files remotely."""

    def __init__(
        self,
        storage_client: storage.Client = storage.Client(),
        local_filepath: str = "/tmp",
        bucket_name: str = "classifai-app-data",
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
        self.bucket_name = bucket_name
        self.bucket_folder = bucket_folder

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
        ext_local_filepath = f"{self.local_filepath}/db"
        for layer1_el in os.listdir(ext_local_filepath):
            if "." in layer1_el:
                all_filepaths_to_copy.append(
                    f"{ext_local_filepath}/{layer1_el}"
                )
            else:
                for layer2_el in os.listdir(
                    f"{ext_local_filepath}/{layer1_el}"
                ):
                    all_filepaths_to_copy.append(
                        f"{ext_local_filepath}/{layer1_el}/{layer2_el}"
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

        bucket = self.storage_client.bucket("classifai-app-data")

        for entry in all_filepaths_to_copy:
            blob = bucket.blob(entry.split(f"{self.local_filepath}/")[1])
            blob.upload_from_filename(entry)

        print("DB files successfully written to GCS bucket.")

    def update(self):
        """Update remote db from local changes."""
        self.delete_existing_gcs_bucket_folder()
        local_db_files = self.list_all_db_files()
        self.write_local_files_to_gcs_bucket(local_db_files)


def get_secret():
    """Access GCP Secret Manager secret value.

    Returns
    -------
    google_api_key : str
        Secret value.
    """

    path = "projects/14177695902/secrets"
    secret = "GOOGLE_API_KEY"  # pragma: allowlist secret
    name = "/".join((path, secret, "versions", "latest"))
    client = SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": name})
    google_api_key = response.payload.data.decode("UTF-8")

    return google_api_key


def pull_vdb_to_local(
    client: storage.Client,
    bucket_name: str = "classifai-app-data",
    prefix: str = "db/",
    local_dir: str = "/tmp/",
    vdb_file: str = "chroma.sqlite3",
):
    """Pull only sqlite3 database / vector store to local /tmp dir.

    Parameters
    ----------
    client : storage.Client
        GCS client object
    bucket_name : str
        GCS bucket name
        Default: 'classifai-app-data'
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

    bucket = client.bucket(bucket_name=bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if filename == vdb_file:
            if not os.path.exists(local_dir + prefix):
                os.mkdir(local_dir + prefix)
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
                        "soc": soc,
                        "description": query_result["documents"][i][j],
                        "distance": query_result["distances"][i][j],
                    }
                    for j, soc in enumerate(query_result["ids"][i])
                ],
            }
            for i, input_id in enumerate(query_result["input_ids"])
        ]
    }

    return processed_result
