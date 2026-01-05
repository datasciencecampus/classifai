import numpy as np
import pandas as pd
import pandera.pandas as pa

##
# Search Input DataClass
##
searchInputSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "query": pa.Column(str),
    },
    coerce=True,
)


class VectorStoreSearchInput(pd.DataFrame):
    _schema = searchInputSchema

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize the class with validated data."""
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreSearchInput":
        """Create a validated VectorStoreSearchInput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreSearchInput":
        """Validate an existing DataFrame against the schema and return a VectorStoreSearchInput."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def query(self) -> pd.Series:
        return self["query"]


##
# Search Output DataClass
##
searchOutputSchema = pa.DataFrameSchema(
    {
        "query_id": pa.Column(str),
        "query_text": pa.Column(str),
        "doc_id": pa.Column(str),
        "rank": pa.Column(int, pa.Check.ge(0)),
        "score": pa.Column(float),
    },
    ordered=True,
    coerce=True,
)


class VectorStoreSearchOutput(pd.DataFrame):
    _schema = searchOutputSchema

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize the class with validated data."""
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreSearchOutput":
        """Create a validated VectorStoreSearchOutput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreSearchOutput":
        """Validate an existing instance against the schema and return a VectorStoreSearchOutput."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def query_id(self) -> pd.Series:
        return self["query_id"]

    @property
    def query_text(self) -> pd.Series:
        return self["query_text"]

    @property
    def doc_id(self) -> pd.Series:
        return self["doc_id"]

    @property
    def doc_text(self) -> pd.Series:
        return self["doc_text"]

    @property
    def rank(self) -> pd.Series:
        return self["rank"]

    @property
    def score(self) -> pd.Series:
        return self["score"]


##
# Reverse Search Input DataClass
##
reverseSearchInputSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "doc_id": pa.Column(str),
    },
    coerce=True,
)


class VectorStoreReverseSearchInput(pd.DataFrame):
    _schema = reverseSearchInputSchema

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
    def text(self) -> pd.Series:
        return self["doc_id"]


##
# Reverse Search Output DataClass
##
reverseSearchOutputSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "doc_id": pa.Column(str),
        "doc_text": pa.Column(str),
    }
)


class VectorStoreReverseSearchOutput(pd.DataFrame):
    _schema = reverseSearchOutputSchema

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize the class with validated data."""
        # Use the from_data logic to validate the input
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = self._schema.validate(df)  # Validate against the schema

        # Call the superclass (pd.DataFrame) constructor
        super().__init__(validated_df)

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "VectorStoreReverseSearchOutput":
        """Create a validated VectorStoreReverseSearchOutput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @classmethod
    def validate(cls, df: pd.DataFrame) -> "VectorStoreReverseSearchOutput":
        """Validate an existing instance against the schema and return a VectorStoreReverseSearchOutputs."""
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def query_id(self) -> pd.Series:
        return self["input_doc_id"]

    @property
    def doc_id(self) -> pd.Series:
        return self["retrieved_doc_id"]

    @property
    def doc_text(self) -> pd.Series:
        return self["doc_text"]


##
# Embed Input DataClass
##
embedInputSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "text": pa.Column(str),
    },
    coerce=True,
)


class VectorStoreEmbedInput(pd.DataFrame):
    _schema = embedInputSchema

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize the class with validated data."""
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


##
# Embed Ouput DataClass
##
embedOutputSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "text": pa.Column(str),
        "embedding": pa.Column(object, pa.Check(lambda x: isinstance(x, np.ndarray), element_wise=True)),
    },
    coerce=True,
)


class VectorStoreEmbedOutput(pd.DataFrame):
    _schema = embedOutputSchema

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
