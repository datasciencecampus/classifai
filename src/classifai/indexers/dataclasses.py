"""Dataclasses for structuring and validating VectorStore input and output data.

This module defines DataFrame-like objects that validate input and output data
for `VectorStore` search, reverse_search, and embedding operations in the
ClassifAI framework. Each class wraps a pandas DataFrame and enforces a schema
using pandera.

Typical usage example:

    search_input = VectorStoreSearchInput({"id": ["1"], "query": ["hello"]})
    embed_input = VectorStoreEmbedInput.from_data({"id": ["1"], "text": ["hello"]})
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa


class VectorStoreSearchInput(pd.DataFrame):
    """DataFrame-like object for forming and validating search query input data.

    This class validates and represents input queries for vector store search.
    Each row contains a unique query identifier and the associated query text.

    Attributes:
        id (pd.Series): Unique identifier for each query.
        query (pd.Series): The query text to search for.
    """

    _schema: pa.DataFrameSchema = pa.DataFrameSchema(
        {
            "id": pa.Column(str),
            "query": pa.Column(str),
        },
        coerce=True,
    )

    def __init__(self, data: dict | pd.DataFrame):
        """Initialise the class with validated data.

        Converts the input data into a DataFrame, validates it against the
        defined schema, then initialises the underlying pd.DataFrame with the
        validated data so this instance can be used as a standard DataFrame.

        Args:
            data: A dictionary or pandas DataFrame containing the input data.
                Must include 'id' and 'query' columns.

        Raises:
            pandera.errors.SchemaError: If the data does not conform to the
                expected schema.
        """
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreSearchInput":
        """Create a validated VectorStoreSearchInput from a dictionary or DataFrame.

        Creates a new instance of VectorStoreSearchInput by validating the
        input data against the defined schema.

        Args:
            data: A dictionary or pandas DataFrame containing the input data.
                Must include 'id' and 'query' columns.

        Returns:
            VectorStoreSearchInput: A validated instance of the class.

        Raises:
            pandera.errors.SchemaError: If the data does not conform to the
                expected schema.
        """
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreSearchInput":
        """Validate an existing DataFrame and return a VectorStoreSearchInput.

        Validates the provided pandas DataFrame against the defined schema
        and returns a new instance of VectorStoreSearchInput if validation
        is successful.

        Args:
            df: A pandas DataFrame to validate. Must include 'id' and 'query'
                columns.

        Returns:
            VectorStoreSearchInput: A validated instance of the class.

        Raises:
            pandera.errors.SchemaError: If the data does not conform to the
                expected schema.
        """
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def query(self) -> pd.Series:
        return self["query"]


class VectorStoreSearchOutput(pd.DataFrame):
    """DataFrame-like object for storing and validating search results.

    This class represents the output of vector store search operations,
    containing query information, matched documents, scores, and similarity
    rankings.

    Attributes:
        query_id (pd.Series): Identifier for the source query.
        query_text (pd.Series): The original query text.
        doc_label (pd.Series): Identifier for the retrieved document.
        doc_text (pd.Series): The text content of the retrieved document.
        rank (pd.Series): The ranking position of the result (0-indexed,
            non-negative).
        score (pd.Series): The similarity score or relevance metric.
    """

    _schema = pa.DataFrameSchema(
        {
            "query_id": pa.Column(str),
            "query_text": pa.Column(str),
            "doc_label": pa.Column(str),
            "doc_text": pa.Column(str),
            "rank": pa.Column(int, pa.Check.ge(0)),
            "score": pa.Column(float),
        },
        ordered=True,
        coerce=True,
    )

    def __init__(self, data: dict | pd.DataFrame):
        """Initialise the class with validated data.

        Converts the input data into a DataFrame, validates it against the
        defined schema, then initialises the underlying pd.DataFrame with the
        validated data so this instance can be used as a standard DataFrame.

        Args:
            data: A dictionary or pandas DataFrame containing the input data.
                Must include 'query_id', 'query_text', 'doc_label', 'doc_text',
                'rank', and 'score' columns.

        Raises:
            pandera.errors.SchemaError: If the data does not conform to the
                expected schema.
        """
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreSearchOutput":
        """Create a validated VectorStoreSearchOutput from a dictionary or DataFrame.

        Creates a new instance of VectorStoreSearchOutput by validating the
        input data against the defined schema.

        Args:
            data: A dictionary or pandas DataFrame containing the input data.
                Must include 'query_id', 'query_text', 'doc_label', 'doc_text',
                'rank', and 'score' columns.

        Returns:
            VectorStoreSearchOutput: A validated instance of the class.

        Raises:
            pandera.errors.SchemaError: If the data does not conform to the
                expected schema.
        """
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreSearchOutput":
        """Validate an existing DataFrame and return a VectorStoreSearchOutput.

        Validates the provided pandas DataFrame against the defined schema
        and returns a new instance of VectorStoreSearchOutput if validation
        is successful.

        Args:
            df: A pandas DataFrame to validate. Must include 'query_id',
                'query_text', 'doc_label', 'doc_text', 'rank', and 'score'
                columns.

        Returns:
            VectorStoreSearchOutput: A validated instance of the class.

        Raises:
            pandera.errors.SchemaError: If the data does not conform to the
                expected schema.
        """
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def query_id(self) -> pd.Series:
        return self["query_id"]

    @property
    def query_text(self) -> pd.Series:
        return self["query_text"]

    @property
    def doc_label(self) -> pd.Series:
        return self["doc_label"]

    @property
    def doc_text(self) -> pd.Series:
        return self["doc_text"]

    @property
    def rank(self) -> pd.Series:
        return self["rank"]

    @property
    def score(self) -> pd.Series:
        return self["score"]


class VectorStoreReverseSearchInput(pd.DataFrame):
    """DataFrame-like object for forming and validating reverse search query input data.

    This class validates and represents input for reverse searches, which find
    similar documents to a given document in the vector store.

    Attributes:
        id (pd.Series): Unique identifier for the reverse search query.
        doc_label (pd.Series): The document ID to find similar documents for.
    """

    _schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str),
            "doc_label": pa.Column(str),
        },
        coerce=True,
    )

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize the class with validated data."""
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreReverseSearchInput":
        """Create a validated VectorStoreReverseSearchInput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreReverseSearchInput":
        """Validate an existing instance against the schema and return a VectorStoreReverseSearchInput."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def doc_label(self) -> pd.Series:
        return self["doc_label"]


class VectorStoreReverseSearchOutput(pd.DataFrame):
    """DataFrame-like object for storing reverse search results.

    This class represents the output of vector store reverse search operations,
    containing knowledgebase examples with the same label as in the query.

    Attributes:
        id (pd.Series): Identifier for the input label for lookup in the
            knowledgebase.
        searched_doc_label (pd.Series): Identifier for the knowledgebase label
            being looked up.
        doc_label (pd.Series): Identifier for the retrieved document with the
            same label.
        doc_text (pd.Series): The text content of the retrieved example.
    """

    _schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str),
            "searched_doc_label": pa.Column(str),
            "doc_label": pa.Column(str),
            "doc_text": pa.Column(str),
        },
        ordered=True,
        coerce=True,
    )

    def __init__(self, data: dict | pd.DataFrame):
        """Initialise the class with validated data."""
        # Use the from_data logic to validate the input
        if (isinstance(data, dict) and not data) or (isinstance(data, pd.DataFrame) and data.empty):
            # If data is empty, create an empty DataFrame with the correct columns
            df = pd.DataFrame(columns=self._schema.columns)
        else:
            df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreReverseSearchOutput":
        """Create a validated VectorStoreReverseSearchOutput from a dictionary or DataFrame."""
        if (isinstance(data, dict) and not data) or (isinstance(data, pd.DataFrame) and data.empty):
            # If data is empty, create an empty DataFrame with the correct columns
            df = pd.DataFrame(columns=cls._schema.columns)
        elif isinstance(data, dict) and len(data["id"]) == 0:
            # If data is an empty DataFrame exported as a dict, create an empty DataFrame with the correct columns
            df = pd.DataFrame(columns=cls._schema.columns)
        else:
            df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreReverseSearchOutput":
        """Validate an existing instance against the schema and return a VectorStoreReverseSearchOutputs."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def searched_doc_label(self) -> pd.Series:
        return self["searched_doc_label"]

    @property
    def doc_label(self) -> pd.Series:
        return self["doc_label"]

    @property
    def doc_text(self) -> pd.Series:
        return self["doc_text"]


class VectorStoreEmbedInput(pd.DataFrame):
    """DataFrame-like object for forming and validating text data to be embedded.

    This class validates and represents input texts that will be converted to
    vector embeddings by the vector store.

    Attributes:
        id (pd.Series): Unique identifier for each text item.
        text (pd.Series): The text content to be embedded.
    """

    _schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str),
            "text": pa.Column(str),
        },
        coerce=True,
    )

    def __init__(self, data: dict | pd.DataFrame):
        """Initialise the class with validated data."""
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreEmbedInput":
        """Create a validated VectorStoreEmbedInput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreEmbedInput":
        """Validate an existing instance against the schema and return a VectorStoreEmbedInput."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def text(self) -> pd.Series:
        return self["text"]


class VectorStoreEmbedOutput(pd.DataFrame):
    """DataFrame-like object for storing and validating embedded vectors and associated metadata.

    This class represents the output of embedding operations, containing the
    original text data along with their computed vector embeddings.

    Attributes:
        id (pd.Series): Unique identifier for each embedded item.
        text (pd.Series): The original text that was embedded.
        embedding (pd.Series): The computed vector embedding (numpy array).
    """

    _schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str),
            "text": pa.Column(str),
            "embedding": pa.Column(object, pa.Check(lambda x: isinstance(x, np.ndarray), element_wise=True)),
        },
        coerce=True,
    )

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize the class with validated data."""
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreEmbedOutput":
        """Create a validated VectorStoreEmbedOutput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreEmbedOutput":
        """Validate an existing instance against the schema and return a VectorStoreEmbedOutput."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def text(self) -> pd.Series:
        return self["text"]

    @property
    def embedding(self) -> pd.Series:
        return self["embedding"]
