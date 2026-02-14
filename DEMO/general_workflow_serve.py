##### first we load the vectoriser used in the vectorstore creation
from classifai.indexers import VectorStore
from classifai.indexers.dataclasses import VectorStoreSearchInput
from classifai.servers.main import run_server
from classifai.vectorisers import HuggingFaceVectoriser

# Our embedding model is pulled down from HuggingFace, or used straight away if previously downloaded
# This also works with many different huggingface models!
vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")


#### now we can load the vectorstore back in without having to create it again
loaded_vectorstore = VectorStore.from_filespace("./DEMO/testdata", vectoriser)


#### look wow! you can search it straight away cause it was loaded back in

search_input_object = VectorStoreSearchInput(
    {
        "id": [42],
        "query": ["a fruit and vegetable farmer"],
    }
)

print(f"Test search {loaded_vectorstore.search(search_input_object, n_results=3)}")


#### and finally, its easy to search your vectorstore via a restAPI service, just run:
run_server([loaded_vectorstore], endpoint_names=["my_vectorstore"])

# Look at https://0.0.0.0:8000/docs to see the Swagger API documentation and test in the browser
