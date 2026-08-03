"""Unit tests for get_router function."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi import APIRouter

from classifai.exceptions import ConfigurationError, DataValidationError
from classifai.indexers import VectorStore
from classifai.servers import get_router


class TestGetRouterInputValidation:
    """Tests for get_router input validation."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create a mocked VectorStore."""
        return Mock(spec=VectorStore)

    def test_get_router_valid_inputs(self, mock_vectorstore):
        """Test successful router creation with valid inputs."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["search_endpoint"]

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)
        assert router is not None

    def test_get_router_vector_stores_must_be_list(self, mock_vectorstore):
        """Test that vector_stores must be a list."""
        vector_stores = (mock_vectorstore,)  # tuple, not list
        endpoint_names = ["search_endpoint"]

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"
        assert "list" in error.message.lower()
        assert error.context["vector_stores_type"] == "tuple"

    def test_get_router_endpoint_names_must_be_list(self, mock_vectorstore):
        """Test that endpoint_names must be a list."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ("search_endpoint",)  # tuple, not list

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"
        assert "list" in error.message.lower()
        assert error.context["endpoint_names_type"] == "tuple"

    def test_get_router_vector_stores_dict_not_list(self, mock_vectorstore):
        """Test that vector_stores cannot be a dict."""
        vector_stores = {"store": mock_vectorstore}
        endpoint_names = ["search_endpoint"]

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"

    def test_get_router_lengths_must_match(self, mock_vectorstore):
        """Test that vector_stores and endpoint_names must have same length."""
        vector_stores = [mock_vectorstore, mock_vectorstore]
        endpoint_names = ["search_endpoint"]  # Only 1 name for 2 stores

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"
        assert "match" in error.message.lower()
        NEXT_ANSWER = 2
        assert error.context["n_vector_stores"] == NEXT_ANSWER
        NEXT_ANSWER = 1
        assert error.context["n_endpoint_names"] == NEXT_ANSWER

    def test_get_router_endpoint_names_must_be_non_empty_strings(self, mock_vectorstore):
        """Test that all endpoint_names must be non-empty strings."""
        vector_stores = [mock_vectorstore]
        endpoint_names = [""]  # Empty string

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"
        assert "non-empty" in error.message.lower() or "empty" in error.message.lower()

    def test_get_router_endpoint_names_no_whitespace_only(self, mock_vectorstore):
        """Test that endpoint_names cannot be whitespace-only."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["   "]  # Whitespace only

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"

    def test_get_router_endpoint_names_must_be_strings(self, mock_vectorstore):
        """Test that all endpoint_names must be strings."""
        vector_stores = [mock_vectorstore]
        endpoint_names = [123]  # Integer, not string

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"

    def test_get_router_endpoint_names_must_be_unique(self, mock_vectorstore):
        """Test that endpoint_names must be unique."""
        vector_stores = [mock_vectorstore, mock_vectorstore]
        endpoint_names = ["search", "search"]  # Duplicate names

        with pytest.raises(DataValidationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "validation_error"
        assert "unique" in error.message.lower()

    def test_get_router_empty_lists(self):
        """Test that empty lists are handled."""
        vector_stores = []
        endpoint_names = []

        # Empty lists are valid - no endpoints to create
        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)


class TestGetRouterVectorStoreValidation:
    """Tests for VectorStore instance validation in get_router."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create a mocked VectorStore."""
        return Mock(spec=VectorStore)

    def test_get_router_each_store_must_be_vectorstore_instance(self, mock_vectorstore):
        """Test that each item must be a VectorStore instance."""
        vector_stores = [mock_vectorstore, "not a vectorstore"]
        endpoint_names = ["store1", "store2"]

        with pytest.raises(ConfigurationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.code == "configuration_error"
        assert "VectorStore" in error.message
        assert error.context["index"] == 1
        assert error.context["vector_store_type"] == "str"

    def test_get_router_invalid_store_at_first_position(self):
        """Test that invalid store at first position raises error."""
        vector_stores = [None]
        endpoint_names = ["store1"]

        with pytest.raises(ConfigurationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.context["index"] == 0

    def test_get_router_invalid_store_at_middle_position(self, mock_vectorstore):
        """Test that invalid store in middle of list is caught."""
        vector_stores = [mock_vectorstore, 123, mock_vectorstore]
        endpoint_names = ["store1", "store2", "store3"]

        with pytest.raises(ConfigurationError) as exc_info:
            get_router(vector_stores, endpoint_names)

        error = exc_info.value
        assert error.context["index"] == 1
        assert error.context["vector_store_type"] == "int"

    def test_get_router_multiple_stores_all_valid(self, mock_vectorstore):
        """Test that multiple valid stores work correctly."""
        mock_store1 = Mock(spec=VectorStore)
        mock_store2 = Mock(spec=VectorStore)
        mock_store3 = Mock(spec=VectorStore)

        vector_stores = [mock_store1, mock_store2, mock_store3]
        endpoint_names = ["store1", "store2", "store3"]

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)


class TestGetRouterRouterCreation:
    """Tests for router creation and endpoint registration."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create a mocked VectorStore."""
        return Mock(spec=VectorStore)

    def test_get_router_returns_apirouter(self, mock_vectorstore):
        """Test that get_router returns an APIRouter instance."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["search"]

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)

    def test_get_router_docs_endpoint_exists(self, mock_vectorstore):
        """Test that the docs endpoint "/" is registered."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["search"]

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        # Check that "/" endpoint is registered
        routes = [route.path for route in router.routes]
        assert "/" in routes

    def test_get_router_docs_endpoint_redirects_to_docs(self, mock_vectorstore):
        """Test that docs endpoint redirects to /docs."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["search"]

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        # Find the docs endpoint and verify it redirects to /docs
        docs_route = None
        for route in router.routes:
            if route.path == "/":
                docs_route = route
                break

        assert docs_route is not None
        assert "GET" in docs_route.methods or "get" in str(docs_route)

    def test_get_router_make_endpoints_called(self, mock_vectorstore):
        """Test that make_endpoints is called with correct arguments."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["search"]

        with patch("classifai.servers.main.make_endpoints") as mock_make:
            router = get_router(vector_stores, endpoint_names)

            # Verify make_endpoints was called with router and dict mapping
            mock_make.assert_called_once()
            call_args = mock_make.call_args
            assert call_args[0][0] == router  # First arg is router
            assert isinstance(call_args[0][1], dict)  # Second arg is dict
            assert "search" in call_args[0][1]  # Dict has endpoint name

    def test_get_router_logging_info_called(self, mock_vectorstore):
        """Test that logging.info is called on router creation."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["search"]

        with patch("classifai.servers.main.logging.info") as mock_log:
            with patch("classifai.servers.main.make_endpoints"):
                router = get_router(vector_stores, endpoint_names)  # noqa: F841

            # Verify logging was called
            assert mock_log.called
            # Find the "Starting ClassifAI Router" log
            log_messages = [call[0][0] for call in mock_log.call_args_list]
            assert any("Router" in msg for msg in log_messages)

    def test_get_router_special_characters_in_names(self, mock_vectorstore):
        """Test that special characters in endpoint names are preserved."""
        vector_stores = [mock_vectorstore, mock_vectorstore]
        endpoint_names = ["search_v1", "search-v2"]

        with patch("classifai.servers.main.make_endpoints") as mock_make:
            router = get_router(vector_stores, endpoint_names)  # noqa: F841

            call_args = mock_make.call_args
            stores_dict = call_args[0][1]
            assert "search_v1" in stores_dict
            assert "search-v2" in stores_dict


class TestGetRouterEdgeCases:
    """Tests for edge cases in get_router."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create a mocked VectorStore."""
        return Mock(spec=VectorStore)

    def test_get_router_single_store_single_endpoint(self, mock_vectorstore):
        """Test with single store and single endpoint."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["only_store"]

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)

    def test_get_router_many_stores(self, mock_vectorstore):
        """Test with many stores and endpoints."""
        mock_stores = [Mock(spec=VectorStore) for _ in range(10)]
        endpoint_names = [f"store_{i}" for i in range(10)]

        with patch("classifai.servers.main.make_endpoints") as mock_make:
            router = get_router(mock_stores, endpoint_names)  # noqa: F841

            # Verify all stores and names are included
            call_args = mock_make.call_args
            stores_dict = call_args[0][1]
            NEXT_ANSWER = 10
            assert len(stores_dict) == NEXT_ANSWER

    def test_get_router_case_sensitive_names(self, mock_vectorstore):
        """Test that endpoint names are case sensitive."""
        mock_store1 = Mock(spec=VectorStore)
        mock_store2 = Mock(spec=VectorStore)

        vector_stores = [mock_store1, mock_store2]
        endpoint_names = ["Store", "store"]  # Different cases

        # Should succeed - they're different names
        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)

    def test_get_router_endpoint_with_unicode_characters(self, mock_vectorstore):
        """Test endpoint names with unicode characters."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["café"]  # Unicode character

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)

    def test_get_router_long_endpoint_names(self, mock_vectorstore):
        """Test with very long endpoint names."""
        vector_stores = [mock_vectorstore]
        endpoint_names = ["a" * 100]  # Very long name

        with patch("classifai.servers.main.make_endpoints"):
            router = get_router(vector_stores, endpoint_names)

        assert isinstance(router, APIRouter)
