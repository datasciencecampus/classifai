"""Unit tests for HuggingFaceVectoriser."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from classifai.exceptions import ConfigurationError, ExternalServiceError, VectorisationError
from classifai.vectorisers import HuggingFaceVectoriser


class TestHuggingFaceVectoriserInitialization:
    """Tests for HuggingFaceVectoriser initialization."""

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_with_valid_model(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test successful initialization with a valid model name."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        vectoriser = HuggingFaceVectoriser("bert-base-uncased")

        assert vectoriser.model_name == "bert-base-uncased"
        assert vectoriser.tokenizer == mock_tokenizer_instance
        assert vectoriser.model == mock_model_instance
        mock_check_deps.assert_called_once_with(["transformers", "torch"], extra="huggingface")

    @patch("classifai.vectorisers.huggingface.check_deps")
    def test_init_missing_dependencies(self, mock_check_deps):
        """Test initialization fails when required dependencies are missing."""
        mock_check_deps.side_effect = ImportError("Missing dependency")

        with pytest.raises(ImportError):
            HuggingFaceVectoriser("bert-base-uncased")

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_invalid_model_name_raises_external_service_error(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test that invalid model name raises ExternalServiceError."""
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(ExternalServiceError) as exc_info:
            HuggingFaceVectoriser("invalid-model-xyz")

        error = exc_info.value
        assert error.code == "external_service_error"
        assert "Failed to load HuggingFace model/tokenizer" in error.message
        assert error.context["vectoriser"] == "huggingface"
        assert error.context["model"] == "invalid-model-xyz"

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_with_custom_tokenizer_kwargs(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test initialization with custom tokenizer kwargs."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        custom_kwargs = {"trust_remote_code": True, "cache_dir": "/custom/path"}
        HuggingFaceVectoriser("bert-base-uncased", tokenizer_kwargs=custom_kwargs)

        # Verify trust_remote_code was preserved (not overridden to False)
        call_kwargs = mock_tokenizer.from_pretrained.call_args[1]
        assert call_kwargs["trust_remote_code"] is True
        assert call_kwargs["cache_dir"] == "/custom/path"

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_trust_remote_code_defaults_to_false(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test that trust_remote_code defaults to False for security."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        HuggingFaceVectoriser("bert-base-uncased")

        tokenizer_call_kwargs = mock_tokenizer.from_pretrained.call_args[1]
        model_call_kwargs = mock_model.from_pretrained.call_args[1]
        assert tokenizer_call_kwargs["trust_remote_code"] is False
        assert model_call_kwargs["trust_remote_code"] is False

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_with_custom_model_revision(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test initialization with custom model revision."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        HuggingFaceVectoriser("bert-base-uncased", model_revision="dev")

        tokenizer_call_kwargs = mock_tokenizer.from_pretrained.call_args[1]
        model_call_kwargs = mock_model.from_pretrained.call_args[1]
        assert tokenizer_call_kwargs["revision"] == "dev"
        assert model_call_kwargs["revision"] == "dev"

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_device_selection_with_explicit_device(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test initialization with explicit device selection."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        with patch("classifai.vectorisers.huggingface.torch") as mock_torch:  # noqa: F841
            mock_device = Mock()
            vectoriser = HuggingFaceVectoriser("bert-base-uncased", device=mock_device)

            assert vectoriser.device == mock_device
            mock_model_instance.to.assert_called_once_with(mock_device)
            mock_model_instance.eval.assert_called_once()

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_device_selection_auto_defaults_to_gpu_if_available(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test that device auto-selection chooses GPU if available."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        with patch("classifai.vectorisers.huggingface.torch") as mock_torch:
            mock_gpu_device = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = mock_gpu_device

            vectoriser = HuggingFaceVectoriser("bert-base-uncased", device=None)

            mock_torch.cuda.is_available.assert_called_once()
            mock_torch.device.assert_called_with("cuda")
            assert vectoriser.device == mock_gpu_device

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_device_selection_fallback_to_cpu(self, mock_model, mock_tokenizer, mock_check_deps):
        """Test that device selection falls back to CPU when GPU unavailable."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        with patch("classifai.vectorisers.huggingface.torch") as mock_torch:
            mock_cpu_device = Mock()
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = mock_cpu_device

            vectoriser = HuggingFaceVectoriser("bert-base-uncased", device=None)

            mock_torch.device.assert_called_with("cpu")
            assert vectoriser.device == mock_cpu_device

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("classifai.vectorisers.huggingface.AutoTokenizer")
    @patch("classifai.vectorisers.huggingface.AutoModel")
    def test_init_device_initialization_failure_raises_configuration_error(
        self, mock_model, mock_tokenizer, mock_check_deps
    ):
        """Test that device initialization failure raises ConfigurationError."""
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.side_effect = RuntimeError("Device not available")

        with patch("classifai.vectorisers.huggingface.torch") as mock_torch:  # noqa: F841
            with pytest.raises(ConfigurationError) as exc_info:
                HuggingFaceVectoriser("bert-base-uncased", device=Mock())

            error = exc_info.value
            assert error.code == "configuration_error"
            assert "Failed to initialise model on device" in error.message
            assert error.context["vectoriser"] == "huggingface"


class TestHuggingFaceVectoriserTransform:
    """Tests for HuggingFaceVectoriser transform method."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Create a mocked HuggingFaceVectoriser instance."""
        with (
            patch("classifai.vectorisers.huggingface.check_deps"),
            patch("classifai.vectorisers.huggingface.AutoTokenizer"),
            patch("classifai.vectorisers.huggingface.AutoModel"),
        ):
            vectoriser = HuggingFaceVectoriser("bert-base-uncased")
            vectoriser.tokenizer = Mock()
            vectoriser.model = Mock()
            vectoriser.device = Mock()
            return vectoriser

    def test_transform_single_string_converts_to_list(self, mock_vectoriser):
        """Test that a single string input is converted to a list."""
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_vectoriser.tokenizer.return_value = mock_inputs

        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_vectoriser.model.return_value = mock_output

        with patch("classifai.vectorisers.huggingface.torch.nn.functional") as mock_f:
            mock_f.normalize.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_vectoriser.transform("hello world")

        call_args = mock_vectoriser.tokenizer.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args == ["hello world"]

    def test_transform_list_of_strings_processes_correctly(self, mock_vectoriser):
        """Test that a list of strings is processed correctly."""
        texts = ["text1", "text2", "text3"]
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_vectoriser.tokenizer.return_value = mock_inputs

        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_vectoriser.model.return_value = mock_output

        with patch("classifai.vectorisers.huggingface.torch.nn.functional") as mock_f:
            mock_f.normalize.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
            result = mock_vectoriser.transform(texts)

        call_args = mock_vectoriser.tokenizer.call_args[0][0]
        assert call_args == texts
        assert isinstance(result, np.ndarray)

    def test_transform_returns_2d_numpy_array(self, mock_vectoriser):
        """Test that transform returns a 2D numpy array."""
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_vectoriser.tokenizer.return_value = mock_inputs

        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_vectoriser.model.return_value = mock_output

        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with patch("classifai.vectorisers.huggingface.torch.nn.functional") as mock_f:
            mock_f.normalize.return_value = embeddings
            result = mock_vectoriser.transform(["text1", "text2"])

        assert isinstance(result, np.ndarray)
        NEXT_ANSWER = 2
        assert result.ndim == NEXT_ANSWER
        assert result.shape == (2, 3)

    def test_transform_output_shape_matches_input_count(self, mock_vectoriser):
        """Test that output shape matches the number of input texts."""
        texts = ["a", "b", "c", "d", "e"]
        embedding_dim = 768

        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_vectoriser.tokenizer.return_value = mock_inputs

        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_vectoriser.model.return_value = mock_output

        embeddings = np.random.rand(len(texts), embedding_dim)
        with patch("classifai.vectorisers.huggingface.torch.nn.functional") as mock_f:
            mock_f.normalize.return_value = embeddings
            result = mock_vectoriser.transform(texts)

        assert result.shape[0] == len(texts)
        assert result.shape[1] == embedding_dim

    def test_transform_tokenization_failure_raises_vectorisation_error(self, mock_vectoriser):
        """Test that tokenization failures raise VectorisationError."""
        mock_vectoriser.tokenizer.side_effect = Exception("Tokenization failed")

        with pytest.raises(VectorisationError) as exc_info:
            mock_vectoriser.transform("some text")

        error = exc_info.value
        assert error.code == "vectorisation_error"
        assert "Tokenization" in error.message or "tokenization" in str(error)

    def test_transform_model_inference_failure_raises_vectorisation_error(self, mock_vectoriser):
        """Test that model inference failures raise VectorisationError."""
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_vectoriser.tokenizer.return_value = mock_inputs
        mock_vectoriser.model.side_effect = Exception("Model inference failed")

        with pytest.raises(VectorisationError) as exc_info:
            mock_vectoriser.transform("some text")

        error = exc_info.value
        assert error.code == "vectorisation_error"

    def test_transform_pooling_failure_raises_vectorisation_error(self, mock_vectoriser):
        """Test that pooling/normalization failures raise VectorisationError."""
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_vectoriser.tokenizer.return_value = mock_inputs

        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_vectoriser.model.return_value = mock_output

        with patch("classifai.vectorisers.huggingface.torch.nn.functional") as mock_f:
            mock_f.normalize.side_effect = Exception("Pooling failed")

            with pytest.raises(VectorisationError) as exc_info:
                mock_vectoriser.transform("some text")

            error = exc_info.value
            assert error.code == "vectorisation_error"

    def test_transform_error_context_includes_model_info(self, mock_vectoriser):
        """Test that error context includes model and vectoriser information."""
        mock_vectoriser.model_name = "distilbert-base-uncased"
        mock_vectoriser.tokenizer.side_effect = Exception("Tokenization failed")

        with pytest.raises(VectorisationError) as exc_info:
            mock_vectoriser.transform("some text")

        error = exc_info.value
        assert error.context["vectoriser"] == "huggingface"
        assert error.context["model"] == "distilbert-base-uncased"
