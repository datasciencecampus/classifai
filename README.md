![ONS Logo](./ONS_Logo_Digital_Colour_Landscape_English_RGB.svg)

# ClassifAI

ClassifAI is a beginner-friendly Python package that simplifies semantic search for classification tasks. It is designed to help developers/users categorize or label new text samples by leveraging a database of previously coded examples. Using embedding and vectorisers, the package creates a vector database of labeled data. When a new, uncategorized text sample is provided, ClassifAI performs a semantic search to find the most similar examples from the database. 

Key Features of the package include:

- Semantic search 
- Use included vectorisers (including GCloud, Huggingface and Ollama) or implement your own 
- Built in support for custom hook logic - write your own custom functions that control the flow of data (spell checking, results deduplication, etc)
- Deploy Easily with FastAPI - Deploy your semantic search classifier with FastAPI capabilities built into the package for easy RestAPI deployment

ClassifAI is ideal for tasks like coding survey responses, classifying free-text data, or building custom classification pipelines with semantic similarity at their core.

ClassifAI facilitates text classification using a semantic seearch over a collection of labelled documents that have been embedded prior to classifying a new text sample:

![ClassifAI Workflow](./vectorstore_search.png)

The most similar samples can then be used to select the correct code for the unlabelled samples by:
- Just choosing the most similar labelled sample,
- Letting some 'human-in-the-loop' choose the correct sample from the top N samples
- Using an LLM model to automatically decide the correct answer from the top N samples
- Any other way you, the user of this package, may choose to do this.


---


## Table of Contents


- [Installation](#installation)
- [Example: Indexing and Searching a Document Collection](#example-indexing-and-searching-a-document-collection)
- [Dev set-up and contributing](#contributing-to-this-repo-and-development-setup)
- [Contact and Support](#contact-and-support)



## Installation

Install the package directly from GitHub in your Python environment

```bash
pip install "git+https://github.com/datasciencecampus/classifAI[huggingface]"
```

## Example: Indexing and searching a knowledgebase

ClassifAI supports statistical classification through searching a knowledgebase. The knowledgebase is a set of labelled examples that show how different texts (e.g. survey responses) map to statistical classifications.
The size and quality of the knowledgebase dictates the quality of results you can expect from using the tool.

#### Step 1: Choose a vectoriser

A vectoriser transforms a query text string into an embedding vector. You can choose from embedding models accessed via HuggingFace, Google or Ollama, or build your own vectoriser.

```python
from classifai.vectorisers import HuggingFaceVectoriser

# Create a vectorizer model
vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")

# You can use the 'transform' method to call the vectoriser
vector = vectoriser.transform("Example text to vectorize")
print(vector.shape)
```

#### Step 2: Build a vector store

You provide a knowledgebase of labelled examples (currently only allows data to be provided as a csv) to build a vector store

```python
from classifai.indexers import VectorStore

vector_store = VectorStore(
    file_name="occupations_knowledgebase.csv",
    vectoriser=vectoriser,
    batch_size=8,
    output_dir="vector_store"
)
```

#### Step 3: Search the vector store

```python
from classifai.indexers.dataclasses import VectorStoreSearchInput

input_data = VectorStoreSearchInput({'id': [1], 'query':["construction worker scaffolder"]})

results = vector_store.search(input_data, n_results=5)
print(results)
```

#### Step 4: Deploy as a REST API

You can use ClassifAI as a local package, or deploy it as an API server, using FastAPI.

```python
from classifai.servers import start_api

start_api(vector_stores=[vector_store], endpoint_names=["Occupations"], port=8000)
```

#### Learn more

Further guides and tutorials can be found in the [DEMO folder](./DEMO/) of this repo. It currently includes the following notebooks:

- [General workflow](./DEMO/general_workflow_demo.ipynb)
- [Custom vectorisers](./DEMO/custom_vectoriser.ipynb)
  - make your own custom vectoriser model that will interact with the core features of the package,
- [Custom pre- and post-processing hooks](./DEMO/custom_preprocessing_and_postprocessing_hooks.ipynb)
  - Add your own custom 'hook' logic to the VectorStore search processes, allowing you to inject custom behaviour to your VectorStores.
- (more demos to come soon)

## Contributing to this repo and development Setup

If you are a developer working on ClassifAI, this section describes how to set up the repo correctly on your local machine to start working on the codebase. 3rd party developers please also read [CONTRIBUTING.md](./CONTRIBUTING.md)

<b>NOTE</b>: This section is for developers who are making alterations and changes to the codebase of this repo itself, not for developers who are using the features of the package.


To begin making changes to the codebase follow the following instructions on your development machine:

1. Clone the repo:
```bash
git clone git@github.com:datasciencecampus/classifAI.git

cd classifAI
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


## Contact and Support

In the event that you come across any issues such as potential bugs in the code or something that may be unclear or not explained in the various DEMO content, please read [CONTRIBUTING.md](./CONTRIBUTING.md) file for instructions. It contains sections on Getting Help, Asking Questions, Reporting Bugs, and other guides on working with the package.
