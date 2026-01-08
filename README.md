# ClassifAI
ClassifAI is a beginner-friendly Python package that simplifies semantic search for classification tasks. It is designed to help developers/users categorize or label new text samples by leveraging a database of previously coded examples. Using embedding and vectorisers, the package creates a vector database of labeled data. When a new, uncategorized text sample is provided, ClassifAI performs a semantic search to find the most similar examples from the database. 

The most similar samples can then be used to select the correct code for the unlabelled samples by:
- Just choosing the most similar labelled sample,
- Letting some 'human-in-the-loop' choose the correct sample from the top N samples
- Using an LLM model to automatically decide the correct answer from the top N samples
- Any other way you, the user of this package, may choose to do this.


ClassifAI is ideal for tasks like coding survey responses, classifying free-text data, or building custom classification pipelines with semantic similarity at their core.

![ONS Logo](./ONS_Logo_Digital_Colour_Landscape_English_RGB.svg)
![ClassifAI ASCII](./classifai.png)

## Table of Contents

- [Feature Overview](#feature-overview)
  - [Vectorising](#1-vectorising)
  - [Indexing](#2-indexing)
  - [Searching](#3-searching)
  - [Using Generative AI Agents](#4-using-generative-ai-agents)
  - [Serving](#5-serving)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Example: Indexing and Searching a Document Collection](#example-indexing-and-searching-a-document-collection)
- [Use Cases and Case Studies](#use-cases-and-case-studies)
- [Using the Package](#using-the-package)
- [Contributing to this Repo and Development Setup](#contributing-to-this-repo-and-development-setup)
- [Contact and Support](#contact-and-support)


## Feature Overview

At its core, ClassifAI is a tool that facilitates text classification using a semantic seearch over a collection of labelled documents that have been embedded prior to classifying a new text sample:


<!-- ### 1. Vectorising
Vectorising is the process of converting text into embeddings (vectors) using a variety of embedding models. These embeddings capture the semantic meaning of the text, enabling efficient similarity searches and comparisons. ClassifAI supports multiple embedding models, allowing users to choose the one that best fits their needs.

---

### 2. Indexing
Indexing involves creating VectorStores from (large) text files. A VectorStore is a structured database of text embeddings that allows for efficient storage, retrieval, and management of vectorized data. This feature is particularly useful for handling large datasets and performing semantic searches.

---

### 3. Searching
Searching allows users to query the VectorStore to retrieve the most relevant results based on semantic similarity. This feature supports advanced search capabilities, enabling users to find information quickly and efficiently.

---

### 5. Serving
Serving makes a VectorStore available through a REST-API, enabling users to perform semantic searches programmatically. This feature allows seamless integration of the VectorStore into other applications or workflows, making it accessible to a broader audience. -->

![ClassifAI Workflow](./vectorstore_search.png)


#### Key Features of the package include:
- Semantic search capabilities,

- Pre-built vectoriser model classes - for converting text to embedding (including GCloud, Huggingface and Ollama support),

- Custom vectoriser model support - create your own Vectoriser model and use it with the ClassifAI framework,

- Built in support for custom hook logic - write your own custom functions that control the flow of data (spell checking, results deduplication, etc),

- Deploy Easily with FastAPI - Deploy your semantic search classifier with FastAPI capabilities built into the package for easy RestAPI deployment.




## Quick Start

#### Installation:

Install the package directly from GitHub in your Python environment

```bash
pip install "git+https://github.com/datasciencecampus/classifAI[huggingface]"
```

---

###Example: Indexing and Searching a Document Collection

#### Step 1. Vectorize Text

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


## Use Cases and Case Studies

There are many possible use cases for this kind of semantic search technology

- Job Classification: Map job descriptions to standardized codes (e.g., ISCO, SOC).
- Document Retrieval: Search for similar documents in large text collections.
- Custom Applications: Build your own classification or retrieval pipelines, deploy with FastAPI, and call it from your own application code.


Also check out the below case studies where ClassifAI has been used

!!!Use cases that can be shared publically should go here>



## Using the package

Reading through the above section on Use Cases and Case Studies is a great way to first determine if the ClassifAI package would be useful for you and your work.
We also provide an extensive set of DEMO jupyter notebooks on how to use the current features of the package.
In particular, we recommend everyone interest in the package reads and tries out the Jupyter Notebook tutorial called 'general_workflow_demo.ipynb'. This notebook introduces all the core features of the pacakge and could be considered the defacto introduction to using the package.:

1. Creating vector embeddings from text using ClassifAI Vectoriser classes we have pre-built for you.
2. How to create a VectorStore and a vector database of your labelled data.
3. How to use a creted VectorStore and search your VectorStore with a new unlabelled sample of data.
4. How to deploy a VectorStore in a restAPI instance so that you can peform classifications/search over a deployed network connection.
5. Advice on what tutorials to look at next, after completing this initial demo.


The /DEMO folder of this repo has a detailed guide of all the different demos/walkthroughs currently avaialable in the package. Beyond the intro workflow notebook described above, it already includes guides on how to:

- make your own custom vectoriser model that will interact with the core features of the package,
- Add your own custom 'hook' logic to the VectorStore search processes, allowing you to inject custom behaviour to your VectorStores.
- (even more demos to come soon)


## Contributing to this repo and development Setup

If you are a developer working on ClassifAI, this section describes how to set up the repo correctly on your local machine to start working on the codebase. (if you are a 3rd party developer looking to contribute please also read through the CONTRIBUTING.md of the repo.)

<b>NOTE</b>: This section is for developers who are making alterations and changes to the codebase of this repo itself, not for developers who are using the features of the package. For those users, check out the above sections: Featuresm Quick Start, and Using the Package.


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

---

## Contact and Support

In the event that you come across any issues such as potential bugs in the code or something that may be unclear or not explained in the various DEMO content, please look into the [CONTRIBUTING.md](./CONTRIBUTING.md) file in the repository. It contains sections on Getting Help, Asking Questions, Reporting Bugs, and other guides on working with the package.
