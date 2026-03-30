"""This module contains a factory for, and example pre-built hooks. Allowing users to manipulate the content of ClassifAI dataclass objects as they enter or leave the `VectorStore`."""

from .default_hooks import CapitalisationStandardisingHook, DeduplicationHook, RagHook
from .hook_factory import HookBase

__all__ = [
    "CapitalisationStandardisingHook",
    "DeduplicationHook",
    "HookBase",
    "RagHook",
]
