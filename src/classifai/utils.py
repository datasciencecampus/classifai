"""ClassifAI utility classes and functions."""

import os

from google.cloud import storage


class DB_Updater:
    """Class of methods to maintain updated DB files remotely."""

    def __init__(
        self,
        storage_client: storage.Client = storage.Client(),
        local_filepath: str = "data/soc-index",
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
