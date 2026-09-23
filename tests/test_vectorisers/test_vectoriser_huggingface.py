"""Unit tests for HuggingFaceVectoriser."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from classifai.exceptions import ConfigurationError, ExternalServiceError, VectorisationError
from classifai.vectorisers import HuggingFaceVectoriser


def make_fake_tokenizer_output(input_ids, attention_mask):
    """Build a fake tokenizer output supporting dict access and .to(device)."""

    class FakeBatchEncoding(dict):
        def to(self, device):
            return self

    return FakeBatchEncoding(
        {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
    )


class TestHuggingFaceVectoriserInitialization:
    """Tests for HuggingFaceVectoriser initialization."""

    @patch("classifai.vectorisers.huggingface.check_deps")
    def test_init_missing_dependencies_raises_error(self, mock_check_deps):
        """Missing torch/transformers should raise the error surfaced by check_deps."""
        mock_check_deps.side_effect = ImportError("torch/transformers not installed")

        with pytest.raises(ImportError):
            HuggingFaceVectoriser(model_name="bert-base-uncased")

        mock_check_deps.assert_called_once_with(["transformers", "torch"], extra="huggingface")

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_valid_model_loads_successfully(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """A valid model name should load tokenizer and model without raising."""
        vectoriser = HuggingFaceVectoriser(model_name="bert-base-uncased")

        assert vectoriser.model_name == "bert-base-uncased"
        assert vectoriser.tokenizer is mock_autotokenizer.from_pretrained.return_value
        assert vectoriser.model is mock_automodel.from_pretrained.return_value

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_invalid_model_name_raises_external_service_error(
        self, mock_autotokenizer, mock_automodel, mock_check_deps
    ):
        """A failure loading the tokenizer/model should raise ExternalServiceError."""
        mock_autotokenizer.from_pretrained.side_effect = OSError("model not found")

        with pytest.raises(ExternalServiceError):
            HuggingFaceVectoriser(model_name="not-a-real-model")

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_device_defaults_to_cpu_when_cuda_unavailable(
        self, mock_autotokenizer, mock_automodel, mock_check_deps
    ):
        """When no device is specified and CUDA is unavailable, device should default to cpu."""
        with patch("torch.cuda.is_available", return_value=False):
            vectoriser = HuggingFaceVectoriser(model_name="bert-base-uncased")

        assert vectoriser.device == torch.device("cpu")

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_device_defaults_to_cuda_when_available(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """When no device is specified and CUDA is available, device should default to cuda."""
        with patch("torch.cuda.is_available", return_value=True):
            vectoriser = HuggingFaceVectoriser(model_name="bert-base-uncased")

        assert vectoriser.device == torch.device("cuda")

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_explicit_device_is_respected(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """An explicitly passed device should be used as-is."""
        vectoriser = HuggingFaceVectoriser(model_name="bert-base-uncased", device="cpu")

        assert vectoriser.device == "cpu"

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_bad_device_raises_configuration_error(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """If placing the model on the device fails, ConfigurationError should be raised."""
        mock_automodel.from_pretrained.return_value.to.side_effect = RuntimeError("invalid device")

        with pytest.raises(ConfigurationError):
            HuggingFaceVectoriser(model_name="bert-base-uncased", device="bad-device")

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_trust_remote_code_defaults_to_false(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """trust_remote_code should default to False for both tokenizer and model kwargs."""
        HuggingFaceVectoriser(model_name="bert-base-uncased")

        _, tok_kwargs = mock_autotokenizer.from_pretrained.call_args
        _, model_kwargs = mock_automodel.from_pretrained.call_args

        assert tok_kwargs["trust_remote_code"] is False
        assert model_kwargs["trust_remote_code"] is False

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_custom_tokenizer_kwargs_passed_through(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """Custom tokenizer_kwargs should be forwarded to AutoTokenizer.from_pretrained."""
        HuggingFaceVectoriser(
            model_name="bert-base-uncased",
            tokenizer_kwargs={"use_fast": False},
        )

        _, call_kwargs = mock_autotokenizer.from_pretrained.call_args
        assert call_kwargs["use_fast"] is False

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_custom_model_kwargs_passed_through(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """Custom model_kwargs should be forwarded to AutoModel.from_pretrained."""
        HuggingFaceVectoriser(
            model_name="bert-base-uncased",
            model_kwargs={"trust_remote_code": True},
        )

        _, call_kwargs = mock_automodel.from_pretrained.call_args
        assert call_kwargs["trust_remote_code"] is True

    @patch("classifai.vectorisers.huggingface.check_deps")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_init_model_revision_passed_through(self, mock_autotokenizer, mock_automodel, mock_check_deps):
        """model_revision should be forwarded to both from_pretrained calls."""
        HuggingFaceVectoriser(model_name="bert-base-uncased", model_revision="v2")

        _, tok_kwargs = mock_autotokenizer.from_pretrained.call_args
        _, model_kwargs = mock_automodel.from_pretrained.call_args

        assert tok_kwargs["revision"] == "v2"
        assert model_kwargs["revision"] == "v2"


class TestHuggingFaceVectoriserTransform:
    """Tests for HuggingFaceVectoriser transform method."""

    @pytest.fixture
    def mock_vectoriser(self):
        """Return a HuggingFaceVectoriser instance with mocked tokenizer/model."""
        with (
            patch("classifai.vectorisers.huggingface.check_deps"),
            patch("transformers.AutoModel"),
            patch("transformers.AutoTokenizer"),
        ):
            vectoriser = HuggingFaceVectoriser(model_name="bert-base-uncased", device="cpu")

            # Replace with controllable mocks for the transform tests.
            vectoriser.tokenizer = Mock()
            vectoriser.model = Mock()

            yield vectoriser

    def _configure_successful_forward_pass(self, vectoriser, batch_size, seq_len, hidden_dim):
        """Wire up tokenizer + model mocks to produce a real small tensor output."""
        vectoriser.tokenizer.return_value = make_fake_tokenizer_output(
            input_ids=[[1] * seq_len] * batch_size,
            attention_mask=[[1] * seq_len] * batch_size,
        )

        fake_model_output = Mock()
        fake_model_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        vectoriser.model.return_value = fake_model_output

    def test_transform_single_string_converts_to_list_and_processes(self, mock_vectoriser):
        """A single string input should be wrapped in a list and produce one embedding row."""
        self._configure_successful_forward_pass(mock_vectoriser, batch_size=1, seq_len=5, hidden_dim=8)

        result = mock_vectoriser.transform("hello world")

        assert result.shape[0] == 1
        call_args, _ = mock_vectoriser.tokenizer.call_args
        assert call_args[0] == ["hello world"]

    def test_transform_list_of_strings_processes_correctly(self, mock_vectoriser):
        """A list of strings should be passed through unchanged."""
        self._configure_successful_forward_pass(mock_vectoriser, batch_size=3, seq_len=6, hidden_dim=8)

        mock_vectoriser.transform(["a", "b", "c"])

        call_args, _ = mock_vectoriser.tokenizer.call_args
        assert call_args[0] == ["a", "b", "c"]

    def test_transform_returns_2d_numpy_array(self, mock_vectoriser):
        """Output should be a 2D numpy array."""
        self._configure_successful_forward_pass(mock_vectoriser, batch_size=2, seq_len=6, hidden_dim=8)

        result = mock_vectoriser.transform(["a", "b"])

        assert result.ndim == 2

    def test_transform_output_shape_matches_input_count(self, mock_vectoriser):
        """Number of output rows should match number of input texts."""
        self._configure_successful_forward_pass(mock_vectoriser, batch_size=4, seq_len=6, hidden_dim=8)

        result = mock_vectoriser.transform(["a", "b", "c", "d"])

        assert result.shape[0] == 4

    def test_transform_tokenisation_failure_raises_vectorisation_error(self, mock_vectoriser):
        """If tokenization raises, it should be wrapped in VectorisationError."""
        mock_vectoriser.tokenizer.side_effect = RuntimeError("bad tokenizer input")

        with pytest.raises(VectorisationError):
            mock_vectoriser.transform(["hello"])

    def test_transform_model_inference_failure_raises_vectorisation_error(self, mock_vectoriser):
        """If the model forward pass raises, it should be wrapped in VectorisationError."""
        mock_vectoriser.tokenizer.return_value = make_fake_tokenizer_output(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
        )
        mock_vectoriser.model.side_effect = RuntimeError("model forward failed")

        with pytest.raises(VectorisationError):
            mock_vectoriser.transform(["hello"])

    def test_transform_pooling_failure_raises_vectorisation_error(self, mock_vectoriser):
        """If pooling fails (e.g. shape mismatch), it should be wrapped in VectorisationError."""
        mock_vectoriser.tokenizer.return_value = make_fake_tokenizer_output(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
        )

        # Mismatched shape between hidden_state (seq_len=5) and attention_mask (seq_len=3)
        # forces a broadcasting error inside the pooling try-block.
        fake_model_output = Mock()
        fake_model_output.last_hidden_state = torch.randn(1, 5, 8)
        mock_vectoriser.model.return_value = fake_model_output

        with pytest.raises(VectorisationError):
            mock_vectoriser.transform(["hello"])
