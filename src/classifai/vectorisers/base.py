"""# Vectoriser Overview.

In our Package, Vectoriser have a simple role:
    - Take in text data (as a string or list of strings)
    - Output numerical embeddings (as a numpy array)
    - Each Vectortiser should provide a `transform` method to perform this conversion.

It is possible for users to implement their own Vectoriser classes by inheriting from the
`VectoriserBase` abstract base class and implementing the `transform` method.


###########################
###########################
# Vectoriser Base Class.

The `VectoriserBase` class provides as the abstract base class for all vectoriser implementations.
It defines the structure and contract that all vectoriser subclasses must adhere to.

Key Responsibilities:
    - Enforce the implementation of a `transform` method in all subclasses.
    - Describes the input parameters and return types of the transform method.
    - Provide a consistent interface for converting text data (as a string or list of strings)
      into numerical embeddings (as a numpy array).

To create a custom vectoriser, users should inherit from `VectoriserBase` and implement the
`transform` method. This method is expected to take text input and return the corresponding
embeddings as a numpy array.

By adhering to this base class, all vectoriser implementations maintain a uniform interface,
making it easier to integrate and switch between different vectoriser implementations.
"""

from abc import ABC, abstractmethod

import numpy as np

##
# The following is the abstract base class for all vectorisers.
##


class VectoriserBase(ABC):
    """Abstract base class for all vectorisers."""

    @abstractmethod
    def transform(self, texts: str | list[str]) -> np.ndarray:
        """Transforms input text(s) into embeddings."""
        pass
