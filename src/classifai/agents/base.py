from abc import ABC, abstractmethod

from classifai.indexers.dataclasses import VectorStoreSearchOutput

##
# The following is the abstract base class for all RAG generative models.
##


class GeneratorBase(ABC):
    """Abstract base class for all Generative RAG models."""

    @abstractmethod
    def transform(
        self,
        results: VectorStoreSearchOutput,
    ) -> VectorStoreSearchOutput:
        """Passes prompt(s) to the generator and returns the generated text(s) and RAG ranking."""
        pass
