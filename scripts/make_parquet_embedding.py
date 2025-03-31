# ruff: noqa
"""This script creates a parquet embedding from a pre-existing Chromadb dataset."""

# %%
import numpy as np
import polars as pl
from google.cloud import storage
from pathlib import Path

from classifai.config import Config
from classifai.utils import pull_vdb_to_local

config = Config("API")
config.setup_logging()

if not config.validate():
    logging.error("Invalid configuration. Exiting.")
    import sys

    sys.exit(1)

# %%

pull_vdb_to_local(
    client=storage.Client(),
    local_dir="data/db/",
    prefix="soc_knowledge_base_db_OLD/",
    bucket_name=config.bucket_name,
)

# %%
client = chromadb.PersistentClient("../data/db/soc_knowledge_base_db_OLD")
collection = client.get_collection(name="classifai-collection")

# %%
embedding = collection.get(include=["metadatas", "documents", "embeddings"])

# # %%
knowledgebase = pl.DataFrame(
    dict(
        ids=embedding.get("ids"),
        documents=embedding.get("documents"),
        embeddings=embedding.get("embeddings"),
        metadatas=embedding.get("metadatas"),
    )
)
knowledgebase = knowledgebase.unnest("metadatas")

# %%
knowledgebase.write_parquet("data/classifai.parquet")
