![ONS Logo](./ONS_Logo_Digital_Colour_Landscape_English_RGB.svg)

# ClassifAI

ClassifAI is a beginner-friendly Python package that simplifies semantic search for classification tasks. It is designed to help developers/users categorize or label new text samples by leveraging a database of previously coded examples. Using embedding and vectorisers, the package creates a vector database of labeled data. When a new, uncategorized text sample is provided, ClassifAI performs a semantic search to find the most similar examples from the database. 

The most similar samples can then be used to select the correct code for the unlabelled samples by:
- Just choosing the most similar labelled sample,
- Letting some 'human-in-the-loop' choose the correct sample from the top N samples
- Using an LLM model to automatically decide the correct answer from the top N samples
- Any other way you, the user of this package, may choose to do this.


ClassifAI is ideal for tasks like coding survey responses, classifying free-text data, or building custom classification pipelines with semantic similarity at their core.

ClassifAI facilitates text classification using a semantic seearch over a collection of labelled documents that have been embedded prior to classifying a new text sample:

![ClassifAI Workflow](./vectorstore_search.png)

Key Features of the package include:

- Semantic search 
- Use included vectorisers (including GCloud, Huggingface and Ollama) or implement your own 
- Built in support for custom hook logic - write your own custom functions that control the flow of data (spell checking, results deduplication, etc)
- Deploy Easily with FastAPI - Deploy your semantic search classifier with FastAPI capabilities built into the package for easy RestAPI deployment

---


## Table of Contents

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Example: Indexing and Searching a Document Collection](#example-indexing-and-searching-a-document-collection)
- [Use Cases and Case Studies](#use-cases-and-case-studies)
- [Using the Package](#using-the-package)
- [Contributing to this Repo and Development Setup](#contributing-to-this-repo-and-development-setup)
- [Contact and Support](#contact-and-support)








## Installation

Install the package directly from GitHub in your Python environment

```bash
pip install "git+https://github.com/datasciencecampus/classifAI[huggingface]"
```

## Example: Indexing and searching a document collection

#### Step 1: Vectorize text

```python
from classifai.vectorisers import HuggingFaceVectoriser

# Create a vectorizer model
vectoriser = HuggingFaceVectoriser(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = vectoriser.transform("Example text to vectorize")
print(vector.shape)
```

#### Step 2: Build a VectorStore

```python
from classifai.indexers import VectorStore

vector_store = VectorStore(
    file_name="data.csv",
    data_type="csv",
    vectoriser=vectoriser,
    batch_size=8,
    output_dir="vector_store"
)
```

#### Step 3: Search the VectorStore

```python
from classifai.indexers.dataclasses import VectorStoreSearchInput

input_data = VectorStoreSearchInput({'id': [1], 'query':["construction worker scaffolder"]})

results = vector_store.search(input_data, n_results=5)
print(results)
```

#### Step 4: Deploy as a REST API

```python
from classifai.servers import start_api

start_api(vector_stores=[vector_store], endpoint_names=["my_data"], port=8000)
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
