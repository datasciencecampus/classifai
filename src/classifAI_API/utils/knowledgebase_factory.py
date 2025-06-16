import logging
import numpy as np
import polars as pl
from google.cloud import storage
from google import genai
from pathlib import Path
from time import sleep
import ollama

from classifAI_API.fast_api.google_configurations.config import Config

def embed_genai(documents: list[str], 
                api_key: str, 
                model_name: str="text-embedding-004", 
                model_task: str="CLASSIFICATION"):

    client = genai.Client(api_key=api_key)
    embeddings = []
    num_chunks = int(np.ceil(len(documents)/100))
    for chunk in range(num_chunks):
        result = client.models.embed_content(
            model=model_name,
            contents=documents[100*chunk:100*(chunk+1)],
            config=genai.types.EmbedContentConfig(task_type=model_task),
        )
        embeddings.extend([doc.values for doc in result.embeddings])
        sleep(2) # rate-limit is 150reqs/minute
    return np.array(embeddings)

def embed_local_LLM(documents: list[str], model_name: str="nomic-embed-text") -> list[float]:
    embeddings = []
    num_chunks = int(np.ceil(len(documents)/100))
    for chunk in range(num_chunks):
        result = ollama.embed(model=model_name, input=documents[100*chunk:100*(chunk+1)])
        embeddings.extend(result.embeddings)
    return np.array(embeddings)

def create_knowledgebase(input_csv_data_filepath="",
                         csv_separator=",",
                         knowledgebase_output_dir=None, 
                         knowledgebase_filename="knowledgebase",
                         local_LLM=None, 
                         all_local=False):


    if all_local:
        config = Config("API", all_local=True, local_LLM=local_LLM)
    else:
        config = Config("API", all_local=False, local_LLM=local_LLM)

    config.setup_logging()
    if not config.all_local and not config.validate():
        logging.error("Invalid configuration. Exiting.")
        import sys
        sys.exit(1)

    try:
        knowledgebase_df = pl.read_csv(input_csv_data_filepath, separator=csv_separator)
    except FileNotFoundError as e:
        logging.error('Input CSV knowledgebase %s not found. Exiting.' %(input_csv_data_filepath))
        import sys
        sys.exit(1)
    except Exception as e:
        logging.error('%s' %(e))
        import sys
        sys.exit(1)

    columns = knowledgebase_df.columns
    if not {"ids", "documents"}.issubset(set(columns)):
        logging.error('ids and documents columns not found in %s' %(input_csv_data_filepath))
        import sys
        sys.exit(1)

    if local_LLM is not None:
        embedding_func = lambda x: embed_local_LLM(x, local_LLM)
    else:
        embedding_func = lambda x: embed_genai(x, 
                                               config.embedding_api_key, 
                                               config.embeddings_model_name, 
                                               config.embeddings_model_task)
    new_embeddings = embedding_func(knowledgebase_df["documents"].to_list())
    knowledgebase_df = knowledgebase_df.with_columns(
        pl.Series(new_embeddings).alias("embeddings")
    )

    if 'label' in columns:
        knowledgebase_df = knowledgebase_df.with_columns(
            pl.col("label").cast(pl.Utf8),
        )
    else:
        knowledgebase_df = knowledgebase_df.with_columns(
            pl.col("documents").alias("label").cast(pl.Utf8),
        )
    out_path = Path(knowledgebase_output_dir) / knowledgebase_filename
    knowledgebase_df.write_parquet(out_path)