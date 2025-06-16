#!/usr/bin/env python

import numpy as np
from classifAI_API.fast_api import app, run_app, setup_app

config, vector_store = setup_app(parquet_filepath_local='./isco_knowledgebase_localLLM.parquet', 
                                 local_LLM='nomic-embed-text',
                                 all_local=True)

@app.post("/new_endpoint_score", description="scoring programmatic endpoint")
def new_endpoint_score(query_embeddings: list[float],
) -> list[float] :
    query_embeddings = np.array(query_embeddings)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings)
    document_embeddings = vector_store.knowledgebase["embeddings"].to_numpy().T
    scores = query_embeddings @ document_embeddings
    return scores.tolist()

run_app()