from abc import ABC, abstractmethod

from classifai.indexers.dataclasses import VectorStoreSearchOutput

##
# The following is the abstract base class for all RAG generative models.
##


class Agentase(ABC):
    """Abstract base class for all Generative and RAG models."""

    @abstractmethod
    def transform(
        self,
        results: VectorStoreSearchOutput,
    ) -> VectorStoreSearchOutput:
        """Passes VectorStoreSearchOutput object, which the Agent manipulates in some way and returns."""
        pass
