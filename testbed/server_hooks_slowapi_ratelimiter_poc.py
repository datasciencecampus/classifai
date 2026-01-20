# /usr/bin/env -S uv run

# ------------- Run ClassifAI ------------- #

# Load packages
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from classifai.indexers import VectorStore
from classifai.servers.main import setup_api, start_api

# Initialise a Vectoriser
from classifai.vectorisers import HuggingFaceVectoriser

vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = VectorStore(
    file_name="fake_soc_dataset.csv",
    data_type="csv",
    vectoriser=vectoriser,
    meta_data=None,
    overwrite=True,
    output_dir="outputs",
)


# Define server hooks:
def test_pre_search_hook(search_endpt_function):
    print("Pre-search hook executed")
    return search_endpt_function


limiter = Limiter(key_func=get_remote_address)
server_hooks = [
    {
        "search": {
            "decorators": [test_pre_search_hook, limiter.limit("2/minute")],
        },
    },
]

app = setup_api(vector_stores=[vector_store], endpoint_names=["fake_soc"], hooks=server_hooks)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

start_api(app, port=8000)
