"""The response format for the SOC classifier."""

from typing import List

from pydantic import BaseModel, Field


class SicSocCandidate(BaseModel):
    """
    Represents a candidate SIC/SOC code based on provided job title and description.

    Attributes
    ----------
        code (str): Plausible SIC/SOC code based on the provided job title and
            description.
        descriptive (str): Descriptive label of the SIC/SOC category associated
            with code.
        likelihood (float): Likelihood of this code with a value between 0 and 100.
    """

    code: str = Field(
        description="Plausible SIC/SOC code based on provided job title and description."
    )
    descriptive: str = Field(
        description="Descriptive label of the SIC/SOC category associated with code."
    )
    likelihood: float = Field(
        description="Likelihood of this code with value between 0 and 100."
    )


class SicSocResponse(BaseModel):
    """Represents a response model for SIC/SOC code assignment.

    Attributes
    ----------
        candidates (List[SicSocCandidate]): List of possible or alternative SIC/SOC
            codes that may be applicable with their descriptive label and estimated
            likelihood.
    """

    candidates: List[SicSocCandidate] = Field(
        description="""List of possible or alternative SIC/SOC codes that may be applicable
        with their descriptive label and estimated likelihood."""
    )
