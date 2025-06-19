# classifAI_API
A Prototype demo of splitting the ClassiF-AI into three different modules that adopt modular responsibility for different parts of the ClassifAI backend functionality

1. <b>Vectorising</b> - Creating Vectors from text with different embedding models
2. <b>Indexing</b> - The process of creating vector stores from (large) text files
3. <b>Serving</b> - Making a VectorStore available through a RESTAPI to search


## Quick Start

`!pip install git+https://github.com/datasciencecampus/classifAI_API@oo-prototype`

#### Given a CSV file with header columns "id,text", a user can execute the following commands:


First create a Vectoriser Model, which allows users to pass text to its <i>.transform()</i> method and get embeddings
```
#Create a vectoriser model
from classifAI_API.vectorisers import HuggingfaceVectoriser
your_vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector = my_vectoriser.transform("ClassifAI is the best classification tool ever!")

print(vector.shape)
print(type(vector))
```

Then the user can pass the vectoriser and a CSV file to a VectorStore constructor:
```
from classifAI_API.indexers import VectorStore

your_vector_store = VectorStore(
    file_name="PATH_TO_YOUR_CSV_FILE.csv>",
    data_type="csv",
    embedder=your_vectoriser,
    batch_size=8
)
```

You can then 'search the vector store on your local system. And its vectors and metadata will be stored in the `classifai_vector_stores` folder

```
your_vector_store.search("your query about your data goes here")

#other statistics about the vector store are available
your_vector_store.num_vectors
your_vector_store.vector_shape
```


Then when you're happy with your VectorStore model, you can start a RESTAPI service:
```
# start the API with the vector store
start_api(vector_stores=[your_vector_store], endpoint_names=["your_data"], port=8000)
```

This will run a restAPI service on your machine and you can find its docs:
`http://localhost:8000`
`htpp://127.0.0.1:8000`



