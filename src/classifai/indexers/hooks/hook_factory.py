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
    """Abstract base class for all hooks requiring customisation.

    Subclasses must implement the __call__ method to define the hook's
    behaviour when applied to VectorStore input or output data.

    Attributes:
        hook_type: A string identifier for the hook type. Defaults to "generic"
            and can be overridden by subclasses or set via kwargs.
        kwargs: Additional keyword arguments passed at initialisation.
    """

    def __init__(self, **kwargs):
        """Sets any attributes required by the hook.

        Args:
            **kwargs: Arbitrary keyword arguments stored on the instance.
                Subclasses may include hook_type to override the default
                value of "generic".
        """
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
        """Defines the behaviour of the hook when called.

        Subclasses must override this method with their processing logic. The
        return type must match the type of the input data.

        Args:
            data: The VectorStore input or output data to process.

        Returns:
            processed_data: The result of processing the input data, which must
                be of the same type as the input data.

        Raises:
            HookError: If the processed data is not of the same type as the
                input data.
        """
        processed_data = data  # Placeholder for processing logic
        if not isinstance(processed_data, type(data)):
            raise HookError(
                f"Processed data must be of the same type as input. "
                f"Expected {type(data).__name__}, got {type(processed_data).__name__}.",
                context={"hook_type": self.hook_type},
            )
        return processed_data
