# %%
"""Notebook to show an example of getting a SOC code without using Langchain."""

import os
import uuid

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from pyprojroot import here

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="", model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=os.path.join(here(), "data/sic-index/db")
)


collection = client.get_or_create_collection(name="my_collection")
collection = client.delete_collection(name="my_collection")
collection = client.create_collection(
    name="my_collection",
    embedding_function=huggingface_ef,
    get_or_create=False,
)
# %%

file_name = "../data/soc-index/soc_title_condensed.txt"
docs = []
code = []
ids = []
if file_name is not None:
    with open(file_name) as file:
        for line in file:
            if line:
                bits = line.split(",", 1)
                docs.append(bits[1])
                code.append(dict(code=bits[0]))
                ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))

collection.add(documents=docs, metadatas=code, ids=ids)
# %%

result = collection.query(
    query_texts=["Doctor", "Professor", "Lawyer"] * 1000,
    n_results=10,
)
# %%
