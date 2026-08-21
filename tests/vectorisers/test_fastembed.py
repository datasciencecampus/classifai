"""Tests for the FastEmbed vectoriser."""

import sys
from types import ModuleType

import numpy as np
import pytest

from classifai.exceptions import ExternalServiceError, VectorisationError
from classifai.vectorisers import fastembed as fastembed_module
from classifai.vectorisers.fastembed import FastEmbedVectoriser


def _install_fake_fastembed(monkeypatch: pytest.MonkeyPatch, text_embedding_cls: type) -> None:
    """Install a fake fastembed module and bypass optional dependency checks."""
    monkeypatch.setattr(fastembed_module, "check_deps", lambda *_args, **_kwargs: None)
    fake_module = ModuleType("fastembed")
    fake_module.TextEmbedding = text_embedding_cls
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)


def test_transform_single_text_returns_two_dimensional_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-string input should still produce a two-dimensional array."""

    class FakeTextEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.array([1.0, 2.0], dtype=np.float32) for _ in texts]

    _install_fake_fastembed(monkeypatch, FakeTextEmbedding)

    vectoriser = FastEmbedVectoriser(model_name="test-model")

    embeddings = vectoriser.transform("hello")

    assert embeddings.shape == (1, 2)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings[0], np.array([1.0, 2.0], dtype=np.float32))


def test_transform_multiple_texts_returns_two_dimensional_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch input should preserve one embedding row per input text."""

    class FakeTextEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.array([index, index + 1], dtype=np.float32) for index, _ in enumerate(texts)]

    _install_fake_fastembed(monkeypatch, FakeTextEmbedding)

    vectoriser = FastEmbedVectoriser(model_name="test-model")

    embeddings = vectoriser.transform(["hello", "world"])

    assert embeddings.shape == (2, 2)
    np.testing.assert_allclose(
        embeddings,
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
    )


def test_initialisation_wraps_model_load_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model construction failures should surface as ExternalServiceError."""

    class FailingTextEmbedding:
        def __init__(self, model_name: str):
            raise RuntimeError(f"cannot load {model_name}")

    _install_fake_fastembed(monkeypatch, FailingTextEmbedding)

    with pytest.raises(ExternalServiceError, match="Failed to load FastEmbed model") as exc_info:
        FastEmbedVectoriser(model_name="broken-model")

    assert exc_info.value.context["vectoriser"] == "fastembed"
    assert exc_info.value.context["model"] == "broken-model"


def test_transform_wraps_embedding_generation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding failures should surface as VectorisationError."""

    class FailingTextEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts: list[str]) -> list[np.ndarray]:
            raise RuntimeError("embedding failed")

    _install_fake_fastembed(monkeypatch, FailingTextEmbedding)

    vectoriser = FastEmbedVectoriser(model_name="test-model")
    texts = ["hello", "world"]

    with pytest.raises(VectorisationError, match="Failed to generate embeddings using FastEmbed") as exc_info:
        vectoriser.transform(texts)

    assert exc_info.value.context["vectoriser"] == "fastembed"
    assert exc_info.value.context["model"] == "test-model"
    assert exc_info.value.context["n_texts"] == len(texts)


def test_transform_empty_input_returns_empty_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input should return an empty two-dimensional float32 array."""

    class FakeTextEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.array([1.0, 2.0], dtype=np.float32) for _ in texts]

    _install_fake_fastembed(monkeypatch, FakeTextEmbedding)

    vectoriser = FastEmbedVectoriser(model_name="test-model")

    embeddings = vectoriser.transform([])

    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32
