import pandas as pd
import pandera as pa
from pandera.typing import Series


class SearchInputSchema(pa.DataFrameModel):
    id: Series[str]  # Ensure the 'id' column is a string
    text: Series[str]  # Ensure the 'text' column is a string

    # Additional metadata columns can be added dynamically if needed
    class Config:
        strict = False  # No additional columns allowed


class SearchOutputSchema(pa.DataFrameModel):
    query_id: Series[str]
    query_text: Series[str]
    doc_id: Series[str]
    doc_text: Series[str]
    rank: Series[int] = pa.Field(ge=0)  # Non-negative integers
    score: Series[float] = pa.Field(ge=0.0, le=1.0)  # Scores between 0 and 1

    # Additional metadata columns can be added dynamically if needed
    class Config:
        strict = False


class ReverseSearchInputSchema(pa.DataFrameModel):
    id: Series[str]  # Ensure the 'id' column is a string
    text: Series[str]  # Ensure the 'text' column is a string

    # Additional metadata columns can be added dynamically if needed
    class Config:
        strict = False


class ReverseSearchOutputSchema(pa.DataFrameModel):
    query_id: Series[str]
    doc_id: Series[str]
    doc_text: Series[str]

    class Config:
        strict = False


class ClassifaiSearchInput(pd.DataFrame):
    _schema = SearchInputSchema

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "ClassifaiSearchInput":
        """Create a validated ClassifaiSearchInput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def text(self) -> pd.Series:
        return self["text"]


class ClassifaiSearchOutput(pd.DataFrame):
    _schema = SearchOutputSchema

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "ClassifaiSearchOutput":
        """Create a validated ClassifaiSearchOutput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
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


class ClassifaiReverseSearchInput(pd.DataFrame):
    _schema = ReverseSearchInputSchema

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "ClassifaiReverseSearchInput":
        """Create a validated ClassifaiReverseSearchInput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)
        return cls(validated_df)

    @property
    def id(self) -> pd.Series:
        return self["id"]

    @property
    def text(self) -> pd.Series:
        return self["text"]


class ClassifaiReverseSearchOutput(pd.DataFrame):
    _schema = ReverseSearchOutputSchema

    @classmethod
    def from_data(cls, data: dict | pd.DataFrame) -> "ClassifaiReverseSearchOutput":
        """Create a validated ClassifaiSearchOutput from a dictionary or DataFrame."""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        validated_df = cls._schema.validate(df)  # Validate against the schema
        return cls(validated_df)

    @property
    def query_id(self) -> pd.Series:
        return self["query_id"]

    @property
    def doc_id(self) -> pd.Series:
        return self["doc_id"]

    @property
    def doc_text(self) -> pd.Series:
        return self["doc_text"]
