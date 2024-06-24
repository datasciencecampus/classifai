"""Notebook demonstarting the embedding module."""
# %%

from classifai.embedding import EmbeddingHandler

embed = EmbeddingHandler(
    k_matches=3, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embed.embed_index(file_name="data/soc-index/soc_title_condensed.txt")
# %%

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
    {
        "id": "3",
        "job_title": "Operational Researcher",
        "company": "Department for Environment, Food and Rural Affairs",
    },
]

result = embed.search_index(
    input_data=input_data,
    id_field="id",
    embedded_fields=["job_title", "company"],
    process_output=True,
)

result
# %%
result = embed.search_index(
    input_data=input_data,
    id_field="id",
    embedded_fields=["job_title"],
    process_output=True,
)

result
# %%
