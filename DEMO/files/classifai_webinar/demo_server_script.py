from classifai.indexers import VectorStore
from classifai.servers import run_server
from classifai.vectorisers import HuggingFaceVectoriser

# Our embedding model is pulled down from HuggingFace, or used straight away if previously downloaded
# This also works with many different huggingface models!
demo_vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")


#### now we can load the vectorstore back in without having to create it again
loaded_vectorstore = VectorStore.from_filespace("./demo_vectorstore", demo_vectoriser)


#### and finally, its easy to search your vectorstore via a restAPI service, just run:
run_server([loaded_vectorstore], endpoint_names=["my_vectorstore"])