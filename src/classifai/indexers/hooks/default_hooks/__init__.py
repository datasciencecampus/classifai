"""Submodule containing the prebuilt hooks for the service."""

from .postprocessing import DeduplicationHook, RagHook
from .preprocessing import CapitalisationStandardisingHook

__all__ = ["CapitalisationStandardisingHook", "DeduplicationHook", "RagHook"]
