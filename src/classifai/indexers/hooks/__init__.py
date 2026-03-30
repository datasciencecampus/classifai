"""Hooks submodule contains factory for hooks and example prebuilt hooks for users."""

from .default_hooks import CapitalisationStandardisingHook, DeduplicationHook, RagHook
from .hook_factory import HookBase

__all__ = [
    "CapitalisationStandardisingHook",
    "DeduplicationHook",
    "HookBase",
    "RagHook",
]
