"""Notebook to show an example of getting a SOC code without using Langchain."""

# %%
import os
import uuid

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import dotenv
import google.generativeai as genai

dotenv.load_dotenv(dotenv.find_dotenv())
gapikey = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gapikey)
google_api_key = gapikey

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=google_api_key,
    task_type="RETRIEVAL_QUERY",
    model_name="models/text-embedding-004",
)

client = chromadb.PersistentClient(path="data/soc-index/db")


collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=huggingface_ef,
)


# %%
def add_documents():
    """Embeds index."""
    file_name = "data/soc-index/soc_title_condensed.txt"
    docs = []
    label = []
    ids = []
    if file_name is not None:
        with open(file_name) as file:
            for line in file:
                if line:
                    bits = line.split(":", 1)
                    docs.append(bits[1].replace("\n", ""))
                    label.append(dict(label=bits[0]))
                    ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))

    collection.add(documents=docs, metadatas=label, ids=ids)


# add_documents()
# %%
id_column = "id"
input_columns = ["job_title", "company"]
input_data = [
    {
        "id": "1",
        "job_title": "Fishing Trawler Captain",
        "company": "Grimsby Fishing Fleet",
    },
    {
        "id": "2",
        "job_title": "Anaesthetist",
        "company": "London General Hospital",
    },
]

query_texts = []
for entry in input_data:
    query_text = []
    for key, item in entry.items():
        if key in input_columns:
            query_text.append(item)
    query_texts.append(" ".join(query_text))

# %%

result = collection.query(
    query_texts=query_texts,
    n_results=2,
    include=["documents", "metadatas", "distances"],
)
# %%
my_list = dict()
for label_list, description_list, distance_list, input_list in zip(
    result["metadatas"], result["documents"], result["distances"], input_data
):
    id = input_list[id_column]
    for label, description, distance in zip(
        label_list, description_list, distance_list
    ):
        label.update({"description": description[:-1]})
        label.update({"distance": distance})
    my_list[input_list[id_column]] = label_list
# %%
result["metadatas"]
# %%
