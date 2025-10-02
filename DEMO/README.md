# Demo for `classifai_package`

This directory contains a set of Jupyter notebooks designed to help you understand and use `classifai_package` effectively.

## Prerequisites

Before running the notebooks, make sure you have `classifai_package` installed. You can install it using pip:

```bash
pip install git+<TODO upon release>
```

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

## Notebooks Overview

This demo includes two Jupyter notebooks:

### 1. `general_workflow_demo.ipynb`

This introduces the core features of `classifai_package`.

It covers:
*   Importing the package and its main components.
*   Initialising a Vectoriser for converting text to vector representation.
*   Creating a VectorStore - a database of labelled examples, and their vector representations, linked with a Vectoriser to allow supplied text to be searched against the vector database using cosine similarity to rank the labelled examples in order of semantic similarity.
*   Creating a FastAPI server to expose the VectorStore's functionality via a REST API.

This notebook is intended for prospective users to get a quick overview of what the package can do, and as a 'jumping off point' for new projects.

### 2. `custom_vectoriser.ipynb`

This notebook demonstrates how to create a new, custom Vectoriser by extending the base `VectoriserBase` class.

It covers:
*   Creating a new `OneHotVectoriser` class, extended from `VectoriserBase`.
*   Setting up a VectorStore which uses the new Vectoriser.

This notebook is for users who want to implement a vectorisation approach not covered by our existing suite of Vectorisers.

---
