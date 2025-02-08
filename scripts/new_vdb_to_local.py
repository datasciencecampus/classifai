"""Refactor of pull_vdb_to_local."""

import os
import shutil
from pathlib import Path

from google.cloud import storage

from classifai.utils import get_secret


def pull_vdb_to_local(
    client: storage.Client,
    bucket_name: str | None = "classifai-app-data",
    prefix: str = "db/",
    local_dir: str = "/tmp/",
    force_refresh: bool = False,
):
    """Pull contents of folder on GCS bucket to local dir.

    Parameters
    ----------
    client : storage.Client
        GCS client object
    bucket_name : str | None
        Name of GCS bucket. If None, fetched from secrets
        Default: None
    prefix : str
        GCS bucket folder
        Default: 'db/'
    local_dir : str
        Location of local/instance temporary directory
        Default: '/tmp/'
    force_refresh : bool
        Whether to delete and re-fetch if database exists
        Default: False

    Examples
    --------
    >>> client = storage.Client()
    >>> pull_vdb_to_local(client, force_refresh=False)
    """
    if bucket_name is None:
        bucket_name = get_secret(
            "APP_DATA_BUCKET", project_id=os.getenv("PROJECT_ID")
        )

    local_path = Path(local_dir)
    target_dir = local_path / prefix

    if target_dir.exists() and not force_refresh:
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = Path(blob.name).relative_to(prefix)
        local_file = target_dir / relative_path
        local_file.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pull GCS folder contents to local directory"
    )
    parser.add_argument("--bucket-name", default="classifai-app-data")
    parser.add_argument(
        "--prefix", default="sic_knowledge_base_db/", help="GCS bucket folder"
    )
    parser.add_argument(
        "--local-dir", default="data/db/", help="Local directory"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh existing files",
    )
    args = parser.parse_args()

    client = storage.Client()
    pull_vdb_to_local(
        client,
        bucket_name=args.bucket_name,
        prefix=args.prefix,
        local_dir=args.local_dir,
        force_refresh=args.force_refresh,
    )
