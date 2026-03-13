# pylint: disable=C0301
"""This module provides functionality for creating a `VectorStore` from a CSV (text)
file.
It defines the `VectorStore` class, which is used to model and create vector databases
from CSV text files using a `Vectoriser` object.

This class requires a `Vectoriser` object from the vectorisers submodule,
to convert the CSV's text data into vector embeddings which are then stored in the
`VectorStore` objects.

Key Features:

  - Batch processing of input files to handle large datasets.
  - Support for CSV file format (additional formats may be added in future updates).
  - Integration with a custom embedder for generating vector embeddings.
  - Logging for tracking progress and handling errors during processing.

VectorStore Class:

  - The `VectorStore` class is initialized with a `Vectoriser` object and a CSV knowledgebase.
  - Additional columns in the CSV may be specified as metadata to be included in the vector database.
  - Upon creation, the `VectorStore` is saved in parquet format for efficient, and quick
    reloading via the `VectorStore`'s `.from_filespace()` method.
  - A new piece of text data (or label) can be queried against the `VectorStore` in the following ways:
    - `.search()`: to find the most semantically similar pieces of text in the vector database.
    - `.reverse_search()`: to find all examples in the knowledgebase that have a given label.
    - `.embed()`: to generate a vector embedding for a given piece of text data.
  - 'Hook' methods may be specified to perform pre-processing on input data before embedding,
    and post-processing on the output of the search methods.
"""

from .dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreEmbedOutput,
    VectorStoreReverseSearchInput,
    VectorStoreReverseSearchOutput,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)
from .hooks import (
    CapitalisationStandardisingHook,
    HookBase,
)
from .main import VectorStore

__all__ = [
    "CapitalisationStandardisingHook",
    "HookBase",
    "VectorStore",
    "VectorStoreEmbedInput",
    "VectorStoreEmbedOutput",
    "VectorStoreReverseSearchInput",
    "VectorStoreReverseSearchOutput",
    "VectorStoreSearchInput",
    "VectorStoreSearchOutput",
]
