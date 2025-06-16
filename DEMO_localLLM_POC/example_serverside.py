#!/usr/bin/env python

import numpy as np
from classifAI_API.fast_api import app, run_app, setup_app

config, vector_store = setup_app(parquet_filepath_local='./isco_knowledgebase_localLLM.parquet', 
                                 local_LLM='nomic-embed-text',
                                 all_local=True)

run_app()