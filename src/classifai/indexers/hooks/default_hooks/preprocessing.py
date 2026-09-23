from classifai.exceptions import HookError
from classifai.indexers.dataclasses import (
    VectorStoreEmbedInput,
    VectorStoreReverseSearchInput,
    VectorStoreSearchInput,
)
from classifai.indexers.hooks.hook_factory import HookBase


class CapitalisationStandardisingHook(HookBase):
    """A pre-processing hook to handle upper-/lower-/sentence-/title-casing.

    Attributes:
        method (str): The method used to apply the selected capitalisation
            transformation to each string value. Must be one of "lower", "upper",
            "sentence", or "title".
        colname (str | list[str]): The column name or list of column names to
            transform.
    """

    def __init__(self, method: str = "lower", colname: str | list[str] = "query"):
        """Sets the specified method for standardising capitalisation.

        Args:
            method (str): Method for standardisation. Must be one of "lower"
                (like this), "upper" (LIKE THIS), "sentence" (Like this), or
                "title" (Like This). Defaults to "lower".
            colname (str | list[str]): The name of one or more fields of the
                Input object to capitalise. Defaults to "query".

        Raises:
            HookError: If method is not one of the accepted values.
        """
        super().__init__(method=method, colname=colname, hook_type="pre_processing")
        if method not in {"lower", "upper", "sentence", "title"}:
            raise HookError(
                "Invalid method for CapitalisationStandardisingHook. "
                "Must be one of 'lower', 'upper', 'sentence', or 'title'.",
                context={self.hook_type: "Capitalisation", "method": method},
            )
        if method == "lower":
            self.method = str.lower
        elif method == "upper":
            self.method = str.upper
        elif method == "sentence":
            self.method = lambda text: text.capitalize() if text else text
        elif method == "title":
            self.method = str.title
        self.colname = colname

    def __call__(
        self, input_data: VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbedInput
    ) -> VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbedInput:
        """Standardises capitalisation in the input data using the set method.

        Args:
            input_data (VectorStoreSearchInput | VectorStoreReverseSearchInput | VectorStoreEmbedInput):
                The VectorStore input data to process.

        Returns:
            A new instance of the same input type with the specified
            column(s) transformed according to the method attribute.

        Raises:
            HookError: If a column name in colname is not found in
                input_data, or if the column is not of string type.
        """
        if isinstance(self.colname, str):
            self.colname = [self.colname]
        for col in self.colname:
            if col not in input_data.columns:
                raise HookError(
                    "Invalid column name passed.", context={"pre_processing": "Capitalisation", "colname": col}
                )
            if col not in input_data.select_dtypes(include=["object"]).columns:
                raise HookError(
                    f"colname is of type {input_data[col].dtype}, expected 'str'.",
                    context={"pre_processing": "Capitalisation", "colname": col},
                )

        processed_input = input_data.copy()
        for col in self.colname:
            processed_input[col] = processed_input[col].apply(self.method)
        # Ensure the processed input still conforms to the schema
        processed_input = input_data.__class__.validate(processed_input)
        return processed_input
