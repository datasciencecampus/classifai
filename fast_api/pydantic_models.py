"""Pydantic Classes to model request and response data for FastAPI RESTful API."""

from fastapi import HTTPException
from pydantic import BaseModel, Field, validator


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
    bridge: str
    description: str
    distance: float
    rank: int


class DeduplicatedEntry(BaseModel):
    """Model deduplicated entries removing bridge and adding score attributes from ResultEntry."""

    label: str
    description: str
    distance: float
    rank: int
    score: float


class ResultsList(BaseModel):
    """model for ranked list of VDB entries for a single row input."""

    input_id: str
    response: list[ResultEntry] | list[DeduplicatedEntry]

    @validator("response")
    def check_zero_length_response(cls, v):
        """Check for empty results list and throw HTTPException response to client."""

        if len(v) == 0:
            raise HTTPException(
                status_code=503,
                detail="Length of response body is zero. FastAPI Backend Fault.",
            )
        return v


class ResultsResponseBody(BaseModel):
    """model for set of ranked lists, for all row entries submmitted."""

    data: list[ResultsList]
    deduplicated_data: list[ResultsList]
