"""The response format for the SOC classifier."""

from typing import List

from pydantic import BaseModel, Field


class SocCandidate(BaseModel):
    """
    Represents a candidate SOC code based on provided job title and description.

    Attributes
    ----------
        soc_code (str): Plausible SOC code based on the provided job title and
            description.
        soc_descriptive (str): Descriptive label of the SOC category associated
            with soc_code.
        likelihood (float): Likelihood of this soc_code with a value between 0 and 100.
    """

    soc_code: str = Field(
        description="Plausible SOC code based on provided job title and description."
    )
    soc_descriptive: str = Field(
        description="Descriptive label of the SOC category associated with soc_code."
    )
    likelihood: float = Field(
        description="Likelihood of this soc_code with value between 0 and 100."
    )


class SocResponse(BaseModel):
    """Represents a response model for SOC code assignment.

    Attributes
    ----------
        soc_candidates (List[SocCandidate]): List of possible or alternative SOC
            codes that may be applicable with their descriptive label and estimated
            likelihood.
    """

    soc_candidates: List[SocCandidate] = Field(
        description="""List of possible or alternative SOC codes that may be applicable
        with their descriptive label and estimated likelihood."""
    )
