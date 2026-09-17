"""Unit tests for VectoriserBase abstract class."""

from __future__ import annotations

import numpy as np
import pytest

from classifai.vectorisers.base import VectoriserBase


class IncompleteVectoriser(VectoriserBase):
    """Subclass that does NOT implement transform - should fail to instantiate."""

    pass


class ConcreteVectoriser(VectoriserBase):
    """Subclass that DOES implement transform - should instantiate fine."""

    def transform(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 4))


class TestVectoriserBaseIsAbstract:
    """Tests that VectoriserBase enforces its abstract interface correctly."""

    def test_cannot_instantiate_base_class_directly(self):
        """VectoriserBase should raise TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            VectoriserBase()

    def test_transform_is_registered_as_abstract_method(self):
        """Transform should be listed in __abstractmethods__."""
        assert "transform" in VectoriserBase.__abstractmethods__

    def test_subclass_missing_transform_cannot_instantiate(self):
        """A subclass that doesn't implement transform should also fail to instantiate."""
        with pytest.raises(TypeError):
            IncompleteVectoriser()

    def test_concrete_subclass_can_be_instantiated(self):
        """A subclass implementing transform should instantiate successfully."""
        vectoriser = ConcreteVectoriser()
        assert isinstance(vectoriser, VectoriserBase)


class TestVectoriserBaseSubclassBehaviour:
    """Sanity checks on how a concrete subclass should behave."""

    def test_concrete_subclass_transform_accepts_single_string(self):
        """Transform should accept a bare string input."""
        vectoriser = ConcreteVectoriser()
        result = vectoriser.transform("hello")

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_concrete_subclass_transform_accepts_list_of_strings(self):
        """Transform should accept a list of strings input."""
        vectoriser = ConcreteVectoriser()
        result = vectoriser.transform(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_concrete_subclass_transform_returns_2d_array(self):
        """The returned array should always be 2-dimensional."""
        vectoriser = ConcreteVectoriser()
        result = vectoriser.transform(["a", "b", "c"])

        assert result.ndim == 2
