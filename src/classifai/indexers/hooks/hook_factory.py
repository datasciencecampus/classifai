from abc import ABC, abstractmethod

from classifai.exceptions import HookError
from classifai.indexers.dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreEmbedOutput,
    VectorStoreReverseSearchInput,
    VectorStoreReverseSearchOutput,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)


class HookBase(ABC):
    """Abstract base class for all post-processing hooks requiring customisation."""

    def __init__(self, **kwargs):
        """Sets any attributes required by the hook."""
        self.hook_type: str = "generic"  # Placeholder for hook type, can be overridden by subclasses
        # or set via kwargs
        self.kwargs = kwargs

    @abstractmethod
    def __call__(
        self,
        data: VectorStoreSearchOutput
        | VectorStoreReverseSearchOutput
        | VectorStoreEmbedOutput
        | VectorStoreSearchInput
        | VectorStoreReverseSearchInput
        | VectorStoreEmbedInput,
    ) -> (
        VectorStoreSearchOutput
        | VectorStoreReverseSearchOutput
        | VectorStoreEmbedOutput
        | VectorStoreSearchInput
        | VectorStoreReverseSearchInput
        | VectorStoreEmbedInput
    ):
        """Defines the behavior of the hook when called."""
        processed_data = data  # Placeholder for processing logic
        if not isinstance(processed_data, type(data)):
            raise HookError(
                f"Processed data must be of the same type as input. "
                f"Expected {type(data).__name__}, got {type(processed_data).__name__}.",
                context={"hook_type": self.hook_type},
            )
        return processed_data
