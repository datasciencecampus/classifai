from classifai.exceptions import HookError
from classifai.indexers.dataclasses import (
    VectorStoreEmbeddingInput,
    VectorStoreReverseSearchInput,
    VectorStoreSearchInput,
)

from .default_hooks import PreProcessingHookBase


class CapitalisationStandardisingHook(PreProcessingHookBase):
    """A pre-processing hook to handle upper-/lower-/sentence-/title-casing."""

    def __init__(self, method: str = "lower"):
        """Inititialises the hook with the specified method for standardising capitalisation.

        Args:
            method (str): Method for standardisation. Must be one of "lower", "upper", "sentence"
                          or "title". Defaults to "lower".
        """
        if self.method not in {"lower", "upper", "sentence", "title"}:
            raise HookError(
                "Invalid method for CapitalisationStandardisingHook. "
                "Must be one of 'lower', 'upper', 'sentence', or 'title'.",
                context={"pre_processing": "Capitalisation", "method": method},
            )
        if method == "lower":
            self.method = str.lower
        elif method == "upper":
            self.method = str.upper
        elif method == "sentence":
            self.method = lambda text: text.capitalize() if text else text
        elif method == "title":
            self.method = str.title

    def __call__(
        self, input_data: VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbeddingInput
    ) -> VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbeddingInput:
        """Standardises capitalisation in the input data as specified by 'method' attribute."""
        processed_input = input_data.copy()
        processed_input["text"] = processed_input["text"].apply(self.method)
        processed_input = processed_input.validate()  # Ensure the processed input still conforms to the schema
        return processed_input
