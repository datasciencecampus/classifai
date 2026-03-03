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


class PreProcessingHookBase(ABC):
    """Abstract base class for all pre-processing hooks requiring customisation."""

    def __init__(self, **kwargs):
        """Sets any attributes required by the hook."""
        self.kwargs = kwargs

    @abstractmethod
    def _setup(self):
        """Performs any setup / initialisation required by the hook."""
        pass

    @abstractmethod
    def __call__(
        self, input_data: VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbedInput
    ) -> VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbedInput:
        """Defines the behavior of the hook when called."""
        processed_input = input_data  # Placeholder for processing logic
        if not isinstance(
            processed_input, (VectorStoreSearchInput, VectorStoreReverseSearchInput, VectorStoreEmbedInput)
        ):
            raise HookError(
                "Output must be an instance of VectorStoreSearchInput, "
                "VectorStoreReverseSearchInput, or VectorStoreEmbedInput.",
                context={"hook_type": "pre_processing"},
            )
        if not isinstance(processed_input, type(input_data)):
            raise HookError(
                f"Processed input must be of the same type as input. "
                f"Expected {type(input_data).__name__}, got {type(processed_input).__name__}.",
                context={"hook_type": "pre_processing"},
            )
        return processed_input


class PostProcessingHookBase(ABC):
    """Abstract base class for all post-processing hooks requiring customisation."""

    def __init__(self, **kwargs):
        """Sets any attributes required by the hook."""
        self.kwargs = kwargs

    @abstractmethod
    def _setup(self):
        """Performs any setup / initialisation required by the hook."""
        pass

    @abstractmethod
    def __call__(
        self, output: VectorStoreSearchOutput | VectorStoreReverseSearchOutput | VectorStoreEmbedOutput
    ) -> VectorStoreSearchOutput | VectorStoreReverseSearchOutput | VectorStoreEmbedOutput:
        """Defines the behavior of the hook when called."""
        processed_output = output  # Placeholder for processing logic
        if not isinstance(
            processed_output, (VectorStoreSearchOutput, VectorStoreReverseSearchOutput, VectorStoreEmbedOutput)
        ):
            raise HookError(
                "Output must be an instance of VectorStoreSearchOutput, "
                "VectorStoreReverseSearchOutput, or VectorStoreEmbedOutput.",
                context={"hook_type": "post_processing"},
            )
        if not isinstance(processed_output, type(output)):
            raise HookError(
                f"Processed output must be of the same type as input. "
                f"Expected {type(output).__name__}, got {type(processed_output).__name__}.",
                context={"hook_type": "post_processing"},
            )
        return processed_output
