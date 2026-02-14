# pylint: disable=C0301
"""This module provides functionality for creating a vector index from a text file.
It defines the `VectorStore` class, which is used to model and create vector databases
from CSV text files using a vectoriser object.

This class interacts with the Vectoriser class from the vectorisers submodule,
expecting that any vector model used to generate embeddings used in the
VectorStore objects is an instance of one of these classes, most notably
that each vectoriser object should have a transform method.

Key Features:
- Batch processing of input files to handle large datasets.
- Support for CSV file format (additional formats may be added in future updates).
- Integration with a custom embedder for generating vector embeddings.
- Logging for tracking progress and handling errors during processing.

Dependencies:
- polars: For handling data in tabular format and saving it as a Parquet file.
- tqdm: For displaying progress bars during batch processing.
- numpy: for vector cosine similarity calculations
- A custom file iterator (`iter_csv`) for reading input files in batches.

Usage:
This module is intended to be used with the Vectoriers mdodule and the
the servers module from ClassifAI, to created scalable, modular, searchable
vector databases from your own text data.
"""

from .dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreEmbedOutput,
    VectorStoreReverseSearchInput,
    VectorStoreReverseSearchOutput,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)
from .main import VectorStore

__all__ = [
    "VectorStore",
    "VectorStoreEmbedInput",
    "VectorStoreEmbedOutput",
    "VectorStoreReverseSearchInput",
    "VectorStoreReverseSearchOutput",
    "VectorStoreSearchInput",
    "VectorStoreSearchOutput",
]
