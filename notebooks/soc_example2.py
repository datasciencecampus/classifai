"""Notebook to show an example of getting a SOC code without using Langchain."""

# %%
import os
import uuid

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import dotenv
import google.generativeai as genai
from pyprojroot import here

dotenv.load_dotenv(dotenv.find_dotenv())
gapikey = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gapikey)
google_api_key = gapikey

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="", model_name="sentence-transformers/all-MiniLM-L6-v2"
)

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=google_api_key, task_type="RETRIEVAL_QUERY"
)

client = chromadb.PersistentClient(
    path=os.path.join(here(), "data/sic-index/db")
)


collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=google_ef,
)


# %%
def add_documents():
    """Embeds index."""
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


# add_documents()
# %%

result = collection.query(
    query_texts=["Doctor", "Professor", "Lawyer"],
    n_results=10,
)
result.get("documents")
# %%
