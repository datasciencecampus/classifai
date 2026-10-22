"""Submodule containing the prebuilt hooks for the service."""

from .postprocessing import CrossEncoderRerankerHook, DeduplicationHook, RagHook
from .preprocessing import CapitalisationStandardisingHook

__all__ = [
    "CapitalisationStandardisingHook",
    "CrossEncoderRerankerHook",
    "DeduplicationHook",
    "RagHook",
]
