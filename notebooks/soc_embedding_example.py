"""Notebook demonstarting the embedding module."""
# %%

from classifai.embedding import EmbeddingHandler

embed = EmbeddingHandler(k_matches=3)

embed.embed_index(file_name="../data/soc-index/soc_title_condensed.txt")
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
]

result = embed.search_index(
    input_data=input_data, id_field="id", process_output=True
)

print(result)
# %%
