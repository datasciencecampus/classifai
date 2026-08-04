from __future__ import annotations

import importlib
import warnings

import classifai


def test_package_version_is_exposed():
    assert classifai.__version__


def test_vectorisers_public_exports_are_importable():
    module = importlib.import_module("classifai.vectorisers")

    assert module.VectoriserBase
    assert module.HuggingFaceVectoriser
    assert module.GcpVectoriser
    assert module.OllamaVectoriser


def test_indexers_public_exports_are_importable():
    module = importlib.import_module("classifai.indexers")

    assert module.VectorStore
    assert module.VectorStoreEmbedInput
    assert module.VectorStoreEmbedOutput
    assert module.VectorStoreSearchInput
    assert module.VectorStoreSearchOutput
    assert module.VectorStoreReverseSearchInput
    assert module.VectorStoreReverseSearchOutput


def test_servers_public_exports_are_importable():
    module = importlib.import_module("classifai.servers")

    assert module.get_router
    assert module.get_server
    assert module.make_endpoints
    assert module.run_server


def test_evaluation_exports_are_importable():
    module = importlib.import_module("classifai.evaluation")

    assert module.Evaluation


def test_evaluation_import_emits_future_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("classifai.evaluation")
        importlib.reload(module)

    assert any(item.category is FutureWarning for item in caught)
