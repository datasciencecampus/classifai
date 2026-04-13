# pylint: disable=C0301
"""Pydantic Classes to model request and response data for ClassifAI FastAPI RESTful API."""

import pandas as pd
from pydantic import BaseModel, Extra, Field, field_validator


class SearchRequestEntry(BaseModel):
    """Atomic model for a single row of `VectorStore` search method input data (i.e. a single query input) , includes `id` and
    `query`.
    """

    id: str = Field(examples=["1"])
    query: str = Field(
        description="User string describing information need/query.",
        examples=["Vegetable farmer"],
    )


class SearchRequestSet(BaseModel):
    """Model for a list of many `SearchRequestEntry` Pydantic models, i.e. several queries to be searched
    in the `VectorStore`.
    """

    entries: list[SearchRequestEntry] = Field(description="array of search queries to be searched in the VectorStore.")

    @field_validator("entries", mode="after")
    @classmethod
    def _unique_entry_ids(cls, entries: list[SearchRequestEntry]) -> list[SearchRequestEntry]:
        ids = [e.id for e in entries]
        seen: set[str] = set()
        dupes: list[str] = []
        for i in ids:
            if i in seen and i not in dupes:
                dupes.append(i)
            seen.add(i)

        if dupes:
            raise ValueError(f"Duplicate entry ids found: {', '.join(dupes)}. Each entry id must be unique.")
        return entries


class SearchResponseEntry(BaseModel):
    """Atomic model for a single row of `VectorStore` search result data (i.e. a single `VectorStore` entry)."""

    doc_label: str = Field(description="The vectorstore row label of the relevant result entry.")
    doc_text: str = Field(description="The vectorstore row text of the relevant result entry.")
    rank: int = Field(description="The rank of the result entry for the given query, with 1 being the most relevant.")
    score: float = Field(description="The similarity score of the result entry for the given query.")

    class Config:
        extra = Extra.allow  # Allow extra keys (e.g., metadata columns)å


class SearchResponseSet(BaseModel):
    """Model for a list of many `SearchResponseEntry` Pydantic models, representing a ranked list of `VectorStore` search results for a provided query."""

    query_id: str = Field(description="The id of the query input for which these are the search results.")
    query_text: str = Field(description="The text of the query input for which these are the search results.")
    entries: list[SearchResponseEntry] = Field(
        description="array of search results for the given query, ranked by relevance to the query."
    )


class SearchResponseBody(BaseModel):
    """Model for the overall search response body, which includes a list of `SearchResponseSet` objects,
    representing the search results for each input query.
    """

    data: list[SearchResponseSet]


class ReverseSearchRequestEntry(BaseModel):
    """Atomic model for a single row of reverse search data includes `id` and `doc_label`."""

    id: str = Field(examples=["1"])
    doc_label: str = Field(
        examples=["101"],
        description="VectorStore row entry label to be looked up, searched in the 'label' column.",
    )


class ReverseSearchRequestSet(BaseModel):
    """Model for a list of many `ReverseSearchRequestEntry` Pydantic models, i.e. several `VectorStore` row entry
    labels to be looked up in the `VectorStore`.
    """

    entries: list[ReverseSearchRequestEntry] = Field(description="array of VectorStore row entry labels to look up.")

    @field_validator("entries", mode="after")
    @classmethod
    def _unique_entry_ids(cls, entries: list[SearchRequestEntry]) -> list[SearchRequestEntry]:
        ids = [e.id for e in entries]
        seen: set[str] = set()
        dupes: list[str] = []
        for i in ids:
            if i in seen and i not in dupes:
                dupes.append(i)
            seen.add(i)

        if dupes:
            raise ValueError(f"Duplicate entry ids found: {', '.join(dupes)}. Each entry id must be unique.")
        return entries


class ReverseSearchResponseEntry(BaseModel):
    """Atomic model for single reverse search result entry, includes retrieved `doc_label` and `doc_text` which
    are expected as str types.
    """

    doc_label: str
    doc_text: str

    class Config:
        extra = Extra.allow  # Allow extra keys (e.g., metadata columns)


class ReverseSearchResponseSet(BaseModel):
    """Model for a list of many `ReverseSearchResponseEntry` pydnatic models, representing a list of `VectorStore`
    entries found (partially) matching an input 'searched_doc_label' and corresponding input `id`.
    """

    input_id: str = Field(
        description="The id of the vectorstore row entry input for which these are the reverse search results."
    )
    searched_doc_label: str = Field(
        description="The vectorstore row entry label that was looked up in the reverse search query."
    )
    entries: list[ReverseSearchResponseEntry] = Field(
        description="array of reverse search results for the given vectorstore row entry, matching (partially) the input doc_label."
    )


class ReverseSearchResponseBody(BaseModel):
    """Model for the overall reverse search response body, which includes a list of `ReverseSearchResponseSet`
    objects, representing the reverse search results for each input `VectorStore` row entry `id`.
    """

    data: list[ReverseSearchResponseSet]


class EmbedRequestEntry(BaseModel):
    """Atomic model for a single text string to be embedded with `VectorStore` embed method with associated `id`."""

    id: str = Field(description="The id of the text entry to be embedded.", examples=["1"])
    text: str = Field(
        description="The text string to be embedded.", examples=["A string to be converted to vector embedding."]
    )


class EmbedRequestSet(BaseModel):
    """Model for a list of many `EmbedRequestEntry` Pydantic models, representing several text strings to be embedded with the `VectorStore` embed method."""

    entries: list[EmbedRequestEntry] = Field(
        description="array of text entries to be embedded, with their corresponding text and id"
    )

    @field_validator("entries", mode="after")
    @classmethod
    def _unique_entry_ids(cls, entries: list[SearchRequestEntry]) -> list[SearchRequestEntry]:
        ids = [e.id for e in entries]
        seen: set[str] = set()
        dupes: list[str] = []
        for i in ids:
            if i in seen and i not in dupes:
                dupes.append(i)
            seen.add(i)

        if dupes:
            raise ValueError(f"Duplicate entry ids found: {', '.join(dupes)}. Each entry id must be unique.")
        return entries


class EmbedResponseEntry(BaseModel):
    """Atomic model for a single embedding result entry, includes `id`, `text` and `embedding`."""

    id: str = Field(description="The id of the text entry that was embedded.")
    text: str = Field(description="The text string that was embedded.")
    embedding: list = Field(
        description="The vector embedding result for the input text string, represented as a list of floats."
    )

    class Config:
        extra = Extra.allow  # Allow extra keys (e.g., metadata columns)


class EmbedResponseBody(BaseModel):
    """Model for set of list of `EmbedResponseEntry` Pydantic objects, for all row entries submmitted to embedding `VectorStore` method."""

    data: list[EmbedResponseEntry] = Field(
        description="array of embedding results, with their corresponding text and id"
    )


def convert_reverse_search_dataframe_to_pydantic_response(
    df: pd.DataFrame, meta_data: dict
) -> ReverseSearchResponseBody:
    """Convert a `VectorStoreReverseSearchOutput` DataFrame into a JSON object conforming to the `ReverseSearchResponseBody` Pydantic
    model.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing reverse search results.
        meta_data (dict): dictionary of metadata column names mapping to their types.

    Returns:
        ReverseSearchResponseBody: Pydantic model containing the API structured result for reverse search `VectorStore` method.
    """
    # identify metadata columns from the DataFrame by checking which columns are in the meta_data dictionary
    hook_columns = (
        set(df.columns)
        .difference(meta_data.keys())
        .difference(
            {
                "id",
                "searched_doc_label",
                "doc_label",
                "doc_text",
            }
        )
    )
    results_list = []

    # Group rows by `id`
    grouped = df.groupby("id")

    for input_id, group_df in grouped:
        # Convert group_df to a list of dictionaries
        rows_as_dicts = group_df.to_dict(orient="records")

        # Build the list of ReverseSearchResponseEntry objects for the current group
        response_entries = []
        for row in rows_as_dicts:
            # Extract metadata columns dynamically
            metadata_values = {meta: row[meta] for meta in meta_data if meta in row}

            # Find other values - added by hooks - any other per-row columns not in reserved/meta
            other_values = {k: v for k, v in row.items() if k in hook_columns}

            # Create a ReverseSearchResponseEntry object
            response_entries.append(
                ReverseSearchResponseEntry(
                    doc_label=row["doc_label"],
                    doc_text=row["doc_text"],
                    **metadata_values,  # Add metadata dynamically
                    **other_values,  # Add any extra columns dynamically
                )
            )

        # Create a ReverseSearchResponseSet object for the current `id` and 'doc_label'
        results_list.append(
            ReverseSearchResponseSet(
                input_id=input_id,
                searched_doc_label=group_df["searched_doc_label"].iloc[
                    0
                ],  # Assuming `doc_label` is the same for all rows in the group
                entries=response_entries,
            )
        )

    # Create the ReverseSearchResponseBody object to be returned
    response_body = ReverseSearchResponseBody(data=results_list)

    return response_body


def convert_search_dataframe_to_pydantic_response(df: pd.DataFrame, meta_data: dict) -> SearchResponseBody:
    """Convert a `VectorStoreSearchOutput` DataFrame into a JSON object conforming to the `SearchResponseBody` Pydantic
    model.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing search results.
        meta_data (dict): dictionary of metadata column names mapping to their types.

    Returns:
        SearchResponseBody: Pydantic model containing the API structured results for search `VectorStore` method.
    """
    # identify metadata columns from the DataFrame by checking which columns are in the meta_data dictionary
    hook_columns = (
        set(df.columns)
        .difference(meta_data.keys())
        .difference(
            {
                "query_id",
                "query_text",
                "doc_label",
                "doc_text",
                "score",
                "rank",
            }
        )
    )

    # Group rows by `query_id`
    grouped = df.groupby("query_id")

    results_list = []
    for query_id, group_df in grouped:
        # Convert group_df to a list of dictionaries
        rows_as_dicts = group_df.to_dict(orient="records")

        # Build the list of SearchResponseEntry objects for the current group
        response_entries = []
        for row in rows_as_dicts:
            # Extract metadata columns dynamically
            metadata_values = {meta: row[meta] for meta in meta_data}

            # Find other values - added by hooks - any other per-row columns not in reserved/meta
            other_values = {k: v for k, v in row.items() if k in hook_columns}

            # Create a SearchResponseEntry object
            response_entries.append(
                SearchResponseEntry(
                    doc_label=row["doc_label"],
                    doc_text=row["doc_text"],
                    rank=row["rank"],  # Assuming `rank` is a column in the DataFrame
                    score=row["score"],  # Assuming `score` is a column in the DataFrame
                    **metadata_values,  # Add metadata dynamically
                    **other_values,  # Add any extra columns dynamically
                )
            )

        # Create a SearchResponseSet object for the current 'query_id' and 'query_text'
        results_list.append(
            SearchResponseSet(
                query_id=query_id,
                query_text=group_df["query_text"].iloc[0],
                entries=response_entries,
            )
        )

    # Create the SearchResponseBody object to be returned
    response_body = SearchResponseBody(data=results_list)

    return response_body


def convert_embedding_dataframe_to_pydantic_response(df: pd.DataFrame) -> EmbedResponseBody:
    """Convert a `VectorStoreEmbedOutput` DataFrame into a JSON object conforming to the `EmbedResponseBody` Pydantic
    model. Unlike the conversion functions for search and reverse search, this function does not take in a `meta_data` dictionary as an argument, as meta data comes from the `VectorStore` which is not accessed during the embedding process, and thus there are no reserved meta data columns to check for. Instead, this function identifies any extra columns in the DataFrame that are not `id`, `text` or `embedding` as "hook" columns, which may have been added by a user with a custom hook attached to the embed method.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing search results.

    Returns:
        EmbedResponseBody: Pydantic model containing the API structured results for embed `VectorStore` method.
    """
    # identify hook columns from the DataFrame by checking which columns are in the required columns
    hook_columns = set(df.columns).difference(
        {
            "id",
            "text",
            "embedding",
        }
    )

    # Build the list of EmbedResponseEntry objects for the current group
    response_entries = []
    for _, row in df.iterrows():
        other_values = {k: v for k, v in row.items() if k in hook_columns}

        # Create an EmbedResponseEntry object
        response_entries.append(
            EmbedResponseEntry(
                id=row["id"],
                text=row["text"],
                embedding=row["embedding"].tolist(),  # Convert numpy array to list for JSON serialization
                **other_values,  # Add any extra columns dynamically
            )
        )
    # Create the EmbedResponseBody object to be returned
    response_body = EmbedResponseBody(data=response_entries)
    return response_body
