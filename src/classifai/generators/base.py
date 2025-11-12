from abc import ABC, abstractmethod

##
# The following is the abstract base class for all RAG generative models.
##


class GeneratorBase(ABC):
    """Abstract base class for all Generative RAG models."""

    @abstractmethod
    def transform(self, prompt: str):
        """Passes prompt(s) to the generator and returns the generated text(s) and RAG ranking."""
        pass
