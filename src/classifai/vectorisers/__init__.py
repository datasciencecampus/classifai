# pylint: disable=C0301
"""This module provides classes for creating and utilizing embedding models from different machine learning libraries and services.
The Vectoriser module offers a unified interface to interact with various other ClassifAI Package Modules.
Generally Vectorisers are used to convert text data into numerical embeddings representations that can be used for
machine learning tasks.

###########################
###########################
# Vectoriser Overview

In our Package, Vectoriser have a simple role:
    - Take in text data (as a string or list of strings)
    - Output numerical dense embeddings (as a numpy array)
    - Each Vectortiser is required to provide a `transform` method to perform this conversion.

It is possible for users to implement their own Vectoriser classes by inheriting from the
`VectoriserBase` abstract base class and implementing the `transform` method.


###########################
###########################
# Implemented Vectorisers

We provide several quick implementations of Vectorisers that interface with popular ML services and libraries.

The module contains the following 'ready-made' classes:
- `GcpVectoriser`: A class for embedding text using Google Cloud Platform's GenAI or Vertex APIs.
- `HuggingFaceVectoriser`: A general wrapper class for Huggingface Transformers
models to generate text embeddings - inlcuding download and instantiation of HF models from the model repositories.
- `OllamaVectoriser`: A general wrapper class for using a locally running ollama
server to generate text embeddings.

Each class is designed to interface with a specific service that provides embedding model
functionality.

The `GcpVectoriser` class leverages Google's GenAI API,

The `HuggingFaceVectoriser` class utilizes models from the Huggingface Transformers library.

The `OllamaVectoriser` class can use any local/downloaded model which can be served by ollama.

These classes abstract the underlying implementation details, providing a simple and consistent
interface for embedding text using different services.

There is an additional demonstation available on how to create a custom vectoriser by overwriting the base
`VectoriserBase` class in the DEMO folder of the codebase GitHub repository.
"""

from .base import VectoriserBase
from .gcp import GcpVectoriser
from .huggingface import HuggingFaceVectoriser
from .ollama import OllamaVectoriser

__all__ = [
    "GcpVectoriser",
    "HuggingFaceVectoriser",
    "OllamaVectoriser",
    "VectoriserBase",
]
