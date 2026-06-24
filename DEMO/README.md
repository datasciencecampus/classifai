# Overview of Demonstrations & Examples

This directory contains a set of Jupyter notebooks designed to help you understand and use `classifai` effectively.

---

## Notebooks Overview

This demo series includes several Jupyter notebooks:

### 1. ✨ ClassifAI Demo - Introduction & Basic Usage ✨ : `general_workflow_demo.ipynb`

This introduces the core features of `classifai`.

It covers:

*   Importing the package and its main components.

*   Initialising a Vectoriser for converting text to vector representation.

*   Creating a VectorStore - a database of labelled examples, and their vector representations, linked with a Vectoriser to allow supplied text to be searched against the vector database using cosine similarity to rank the labelled examples in order of semantic similarity.

*   Creating a FastAPI server to expose the VectorStore's functionality via a REST API.

This notebook is intended for prospective users to get a quick overview of what the package can do, and as a 'jumping off point' for new projects.

### 2. Creating Your Own Vectoriser : `custom_vectoriser.ipynb`

This notebook demonstrates how to create a new, custom Vectoriser by extending the base `VectoriserBase` class.

It covers:

*   Creating a new `OneHotVectoriser` class, extended from `VectoriserBase`.

*   Setting up a VectorStore which uses the new Vectoriser.

This notebook is for users who want to implement a vectorisation approach not covered by our existing suite of Vectorisers.

### 3. VectorStore pre- and post- processing logic with _Hooks_ 🪝 : `using_hooks.ipynb`

This notebook demostrates how to add custom Python code logic to the VectorStore search pipeline, such as performing spell checking on user input, without breaking the data flow of the ClassifAI VectorStore.

It covers:

* How the VectorStore handles input data and output data

* How to write 'hooks' - Python functions that can manipulate the input and output data of the different VectorStore methods

* The Hook base class, which can be used to create your own hooks.

* Pre-made hooks that come read-made with ClassifAI that you can import straight away.

* How to ensure your hooks don't break the dataflow by following the required input and output dataclasses

* Examples of different kinds of hooks that can be written - [spellchecking, deduplicating results, adding extra info to results based on result ids]

### 4. AI agents in ClassifAI for classification and other tasks ✨ : `ai_agent_hooks.ipynb`

This notebooks describes how Generative AI Agents can be used with the VectorStore for different tasks such as classification, utilising Hooks and showcasing a pre-made hook class - RagHook.

It covers:

* A general introduction to Classification in ClassifAI and the advantages of using Generative AI to achieve this.

* An overview of Retrieval Augmented Generation RAG and how this operates with ClassifAI Hooks and search results.

* A showcase of the pre-made `RagHook` class and performing different RAG type tasks on `VectorStoreSearchOutput` results including classification, reranking and keyword identication, and how to customise the specific task using this class.

### 5. Evaluating VectorStore Performance with Metrics : `evaluation_workflow_demo.ipynb`

This notebook demonstrates how to use the Evaluation module to assess the performance of one or more VectorStore instances against ground-truth labelled data.

It covers:

* An introduction to the Evaluation module and its multi-class, single-label classification focus.

* The available evaluation metrics

* Creating multiple VectorStore instances with varying data coverage to showcase performance differences.

* Instantiating an `Evaluation` object with ground truth data and selected metrics.

* Running the `evaluate()` method to compute metrics across multiple VectorStores.

* Memory-efficient evaluation using callable functions to load VectorStores on-demand, useful when evaluating many or large VectorStores.

**Note:** The Evaluation module is currently in development and its API is subject to change in future releases.
---

## Installation of classifai

#### *0)* [optional] Create and activate a virtual environment from the command line

##### Using pip + venv

Create a virtual environment:

`python -m venv .venv`

##### Using UV

Create a virtual environment:

`uv venv`

##### Activating your environment

(macOS / Linux):

`source .venv/bin/activate`

Activate it (Windows):

`source .venv/Scripts/activate`

#### *1)* Install the classifai package

##### Using pip

`pip install "https://github.com/datasciencecampus/classifai/releases/download/v1.0.0/classifai-1.0.0-py3-none-any.whl"`

##### Using uv

one-off:

`uv pip install "https://github.com/datasciencecampus/classifai/releases/download/v1.0.0/classifai-1.0.0-py3-none-any.whl"`

add as project dependency:

`uv add "https://github.com/datasciencecampus/classifai/releases/download/v1.0.0/classifai-1.0.0-py3-none-any.whl"`


#### *2)* Install optional dependencies

##### Using pip

`pip install "classifai[<dependency>]"` 

where `<dependency>` is one or more of `huggingface`,`gcp`,`ollama`, or `all` to install all of them.
##### Using uv

one-off installation

`uv pip install "classifai[<dependency>]"`

add as project dependency

`uv add "classifai[<dependency>]"`

---

## Prerequisites

You may wish to download each notebook individually and the demo dataset individually - each notebook contains specific installation instructions on how to set up an environemnt and download the package

## Running the Demo

To start the demo, launch Jupyter Notebook or JupyterLab from your terminal in this directory:

```bash
jupyter notebook
```

Or, if you prefer JupyterLab:

```bash
jupyter lab
```

Then, open the notebooks in your browser. 
We recommend going through the the `general_workflow_demo.ipynb` notebook for a broad overview of the package before moving onto the `custom_vectoriser.ipynb` notebook, which covers a more advanced use-case.