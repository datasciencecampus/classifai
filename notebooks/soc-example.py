"""Notebook to show an example of getting a SOC code."""

# %%
from classifAI.embedding import EmbeddingHandler

####  test embedding on handfull examples
# %%
embed = EmbeddingHandler(embedding_model_name="all-MiniLM-L6-v2")
# "all-MiniLM-L6-v2" "all-mpnet-base-v2" "textembedding-gecko@003"
# %%
embed.embed_index(
    from_empty=True, file_name="../data/soc-index/soc_title_condensed.txt"
)
# %%
for ent in embed.search_index(query="Data Scientist"):
    print(f"{ent['code']}: {ent['title'][:-1]}, Distance: {ent['distance']}")

# %%
