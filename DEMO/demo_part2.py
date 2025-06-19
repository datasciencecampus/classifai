####
# THIS IS A SECOND PART OF THE DEMO THAT SHOWS HOW TO LOAD
# IN A VECTOR STORE YOU'VE CREATED AND THEN HOW TO START AN API SERVICE
# # IT ASSUMES YOU HAVE ALREADY RUN THE FIRST PART OF THE DEMO
# Check out part one at demo_part_2 


# To run just execute this file in your terminal:
# python demo_part2.py
# or
# uv run demo_part2.py
####


from classifAI_API.indexers import VectorStore
from classifAI_API.servers import start_api
from classifAI_API.vectorisers import GcpVectoriser, HuggingFaceVectoriser

# reinitialise your vector model
your_vectoriser = HuggingFaceVectoriser(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  #or initialse the vector model you used
)

# loads the vector store back in without having to re-embed the data
your_vector_store = VectorStore.from_filespace(
    folder_path="classifai_vector_stores/testdata", #make sure you pass the right path
    vectoriser=your_vectoriser,
)


# start the API with the vector store on port 8000
start_api(vector_stores=[your_vector_store], endpoint_names=["test_dataset"], port=8000)
