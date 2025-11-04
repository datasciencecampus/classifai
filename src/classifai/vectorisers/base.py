"""This module provides classes for creating and utilizing embedding models from different services.
The Vectoriser module offers a unified interface to interact with various other ClassifAI Package Modules.
Generally Vectorisers are used to convert text data into numerical embeddings that can be used for
machine learning tasks.

###########################
###########################
# Vectoriser Overview

In our Package, Vectoriser have a simple role:
    - Take in text data (as a string or list of strings)
    - Output numerical embeddings (as a numpy array)
    - Each Vectortiser should provide a `transform` method to perform this conversion.

It is possible for users to implement their own Vectoriser classes by inheriting from the
`VectoriserBase` abstract base class and implementing the `transform` method.


###########################
###########################
# Implemented Vectorisers

We provide several quick implementations of Vectorisers that interface with popular services and libraries.

The module contains the following 'ready-made' classes:
- `GcpVectoriser`: A class for embedding text using Google Cloud Platform's GenAI API.
- `HuggingFaceVectoriser`: A general wrapper class for Huggingface Transformers
models to generate text embeddings.
- `OllamaVectoriser`: A general wrapper class for using a locally running ollama
server to generate text embeddings.

Each class is designed to interface with a specific service that provides embedding model
functionality.

The `GcpVectoriser` class leverages Google's GenAI API,

The `HuggingFaceVectoriser` class utilizes models from the Huggingface Transformers library.

The `OllamaVectoriser` class can use any local/downloaded model which can be served by ollama.

These classes abstract the underlying implementation details, providing a simple and consistent
interface for embedding text using different services.
"""

from abc import ABC, abstractmethod

import numpy as np

##
# The following is the abstract base class for all vectorisers.
##


class VectoriserBase(ABC):
    """Abstract base class for all vectorisers."""

    @abstractmethod
    def transform(self, texts: str | list[str]) -> np.ndarray:
        """Transforms input text(s) into embeddings."""
        pass
