"""This module contains a factory for, and example pre-built hooks. Allowing users to manipulate the content of ClassifAI dataclass objects as they enter or leave the VectorStore.

Hooks are callables applied to the input/output dataclasses of VectorStore methods (e.g. search, reverse_search, embed) to implement pre- and post-processing. A hook must accept a single dataclass instance and return a valid instance of the same type, enabling chaining for tasks like text standardisation, input sanitisation, result filtering/aggregation, and metadata injection.

This submodule exposes HookBase for building configurable hooks, plus a set of default hooks for common workflows.
"""

from .default_hooks import (
    CapitalisationStandardisingHook,
    DeduplicationHook,
    HuggingFaceRagHook,
    RagHook,
)
from .hook_factory import HookBase

__all__ = [
    "CapitalisationStandardisingHook",
    "DeduplicationHook",
    "HookBase",
    "HuggingFaceRagHook",
    "RagHook",
]
