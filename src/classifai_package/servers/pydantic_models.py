"""Pydantic Classes to model request and response data for FastAPI RESTful API."""

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Extra, Field, validator


class ClassifaiEntry(BaseModel):
    """Model for a single row of data (SOC or SIC row etc), includes 'id' and 'description' which are expected as str type."""

    id: str = Field(examples=["1"])
    description: str = Field(
        description="User string describing occupation or industry",
        examples=["A butcher's shop"],
    )


class ClassifaiData(BaseModel):
    """Pydantic object which contains list of many SOC/SIC Classifai Entry pydantic models."""

    entries: list[ClassifaiEntry] = Field(
        description="array of SOC/SIC Entries to be classified"
    )


class ResultEntry(BaseModel):
    """Model for single vdb entry."""

    label: str
    description: str
    score: float
    rank: int

    class Config:
        extra = Extra.allow  # Allow extra keys (e.g., metadata columns)


class ResultsList(BaseModel):
    """model for ranked list of VDB entries for a single row input."""

    input_id: str
    response: list[ResultEntry]


class ResultsResponseBody(BaseModel):
    """model for set of ranked lists, for all row entries submmitted."""

    data: list[ResultsList]


class EmbeddingsList(BaseModel):
    """model for set of embeddings lists, for all row entries submmitted."""

    idx: str
    description: str
    embedding: list


class EmbeddingsResponseBody(BaseModel):
    """model for set of list of embeddings, for all row entries submmitted."""

    data: list[EmbeddingsList]


def convert_dataframe_to_pydantic_response(
    df: pd.DataFrame, meta_data: dict
) -> ResultsResponseBody:
    """Convert a Pandas DataFrame into a JSON object conforming to the ResultsResponseBody Pydantic model.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing query results.
        meta_data (dict): dictionary of metadata column names mapping to their types.

    Returns:
        ResultsResponseBody: Pydantic model containing the structured response.
    """
    # Group rows by `query_id`
    grouped = df.groupby("query_id")

    results_list = []

    for query_id, group_df in grouped:
        # Convert group_df to a list of dictionaries
        rows_as_dicts = group_df.to_dict(orient="records")

        # Build the list of ResultEntry objects for the current group
        response_entries = []
        for row in rows_as_dicts:
            # Extract metadata columns dynamically
            metadata_values = {meta: row[meta] for meta in meta_data.keys()}

            # Create a ResultEntry object
            response_entries.append(
                ResultEntry(
                    label=row["doc_id"],
                    description=row["doc_text"],
                    score=row["score"],  # Assuming `score` is a column in the DataFrame
                    rank=row["rank"],  # Assuming `rank` is a column in the DataFrame
                    **metadata_values,  # Add metadata dynamically
                )
            )

        # Create a ResultsList object for the current query_id
        results_list.append(
            ResultsList(
                input_id=query_id,
                response=response_entries,
            )
        )

    # Create the ResultsResponseBody object
    response_body = ResultsResponseBody(data=results_list)

    return response_body
