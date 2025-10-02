# classifAI_package
A generalised, extendable and modular solution for LLM automated / assisted classification.

The main features offered by the package are;
1. <b>Vectorising</b> - Creating text embeddings (vectors) from text using a variety of embedding models
2. <b>Indexing</b> - The process of creating VectorStores from (large) text files
3. <b>Serving</b> - Making a VectorStore available through a REST-API to search


## Quick Start

#### Installation:
`pip install git+https://github.com/datasciencecampus/classifAI_package`

#### Given a CSV file with header columns "id, text", a user can execute the following commands:

First create a Vectoriser Model, which allows users to pass text to its `.transform()` method to convert the text to a vector.
```python

#Create a vectoriser model
from classifai_package.vectorisers import HuggingfaceVectoriser
your_vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector = your_vectoriser.transform("ClassifAI_package is the best classification tool ever!")

print(vector.shape, type(vector))
```

Then pass the vectoriser and a CSV file to a VectorStore constructor to build a vector database that you can interact with through the class.
```python
from classifai_package.indexers import VectorStore

your_vector_store = VectorStore(
    file_name="<PATH_TO_YOUR_CSV_FILE>.csv",
    data_type="csv",
    embedder=your_vectoriser,
    batch_size=8,
    meta_data={'extra_column_1': int, 'extra_column_2': str},
    output_dir="my_vector_store"
)
```
You can 'search' the VectorStore on your local system. 
```python
your_vector_store.search("your query about your data goes here", n_results=5)

#other statistics about the vector store are available
your_vector_store.num_vectors
your_vector_store.vector_shape
```
The vectors and metadata will be stored in the `my_vector_store/` folder, to be quickly reloaded later.
```python
reloaded_vector_store = VectorStore.from_filespace('my_vector_store', your_vectoriser)
reloaded_vector_store.search("your query about your data goes here")
```

When you're happy with your VectorStore model, you can start a REST-API service:
```python
from classifai_package.servers import start_api

start_api(vector_stores=[your_vector_store], endpoint_names=["your_data"], port=8000)
```

This will run a FastAPI based REST-API service on your machine and you can find its docs:

`http://localhost:8000`

`htpp://127.0.0.1:8000`

---

## Development Setup

1. Clone the repo:
```bash
git clone git@github.com:datasciencecampus/classifAI_package.git

cd classifAI_package
```

2. Set up pre-commit hooks:
```bash
make setup-git-hooks
```
  (or, if you don't have Docker available)
```bash
make setup-git-hooks-no-docker
```

3. Create / activate the virtual environment:
```bash
uv lock

uv sync
```

And that's you good to go!

During development, you might want to run linters / code vulnerability scans; you can do so at any point via
```bash
make check-python
```
---