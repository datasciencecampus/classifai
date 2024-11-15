"""Functions for downloading CSV and Excel files from GCP."""

import io
from typing import Optional

import pandas as pd
from google.cloud import storage


def download_csv_from_gcp(
    bucket_name: str, blob_name: str, file_path: str
) -> pd.DataFrame:
    """Download and read CSV file from Google Cloud Storage.

    Args:
        bucket_name: Name of the GCP bucket
        blob_name: Path to the file within the bucket
        file_path: Local path where to save the file

    Returns
    -------
        DataFrame containing the CSV data

    Raises
    ------
        Exception: If file download or reading fails
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download as bytes
        file_bytes = blob.download_as_bytes()

        # Read directly from memory
        df = pd.read_csv(io.BytesIO(file_bytes))

        return df

    except Exception as e:
        raise Exception(
            f"Failed to download/read CSV from GCS: {str(e)}"
        ) from e


def download_excel_from_gcp(
    bucket_name: str, blob_name: str, sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """Download and read Excel file from Google Cloud Storage.

    Args:
        bucket_name: Name of the GCP bucket
        blob_name: Path to the file within the bucket
        sheet_name: Name of the Excel sheet to read (optional)

    Returns
    -------
        DataFrame containing the Excel data

    Raises
    ------
        Exception: If file download or reading fails
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download as bytes
        file_bytes = blob.download_as_bytes()

        # Read Excel with explicit engine
        if sheet_name:
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                engine="openpyxl",
                sheet_name=sheet_name,
            )
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

        return df

    except Exception as e:
        raise Exception(
            f"Failed to download/read Excel from GCS: {str(e)}"
        ) from e
