"""A module that provides a wrapper for Huggingface Transformers models to generate text embeddings."""

from classifai._optional import check_deps
from classifai.exceptions import ConfigurationError, ExternalServiceError, VectorisationError

from .base import VectoriserBase


class HuggingFaceVectoriser(VectoriserBase):
    """A general wrapper class for Huggingface Transformers models to generate text embeddings.

    Attributes:
        model_name (str): The name of the Huggingface model to use.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the specified model.
        model (transformers.PreTrainedModel): The Huggingface model instance.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
    """

    def __init__(self, model_name, device=None, model_revision="main"):
        """Initializes the HuggingfaceVectoriser with the specified model name and device.

        Args:
            model_name (str): The name of the Huggingface model to use.
            device (torch.device, optional): The device to use for computation. Defaults to GPU if available, otherwise CPU.
            model_revision (str, optional): The specific model revision to use. Defaults to "main".

        Raises:
            ExternalServiceError: If the model or tokenizer cannot be loaded.
            ConfigurationError: If the model cannot be initialized on the specified device.
        """
        check_deps(["transformers", "torch"], extra="huggingface")
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)  # nosec: B615
            self.model = AutoModel.from_pretrained(model_name, revision=model_revision)  # nosec: B615
        except Exception as e:
            raise ExternalServiceError(
                "Failed to load HuggingFace model/tokenizer.",
                context={"vectoriser": "huggingface", "model": model_name, "revision": model_revision},
            ) from e

        # Device selection / model placement is local configuration/runtime.
        try:
            if device is not None:
                self.device = device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize model on device.",
                context={"vectoriser": "huggingface", "model": model_name, "device": str(device) if device else "auto"},
            ) from e

    def transform(self, texts):
        """Transforms input text(s) into embeddings using the Huggingface model.

        Args:
            texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.

        Raises:
            VectorisationError: If tokenization, model inference, or embedding extraction fails.
        """
        import torch  # type: ignore

        # If a single string is passed as arg to texts, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenization / tensor move can fail (e.g., device issues, weird tokenizer config)
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        except Exception as e:
            raise VectorisationError(
                "Tokenization failed.",
                context={"vectoriser": "huggingface", "model": self.model_name, "n_texts": len(texts)},
            ) from e

        # Forward pass can fail (OOM, dtype/device mismatch, model bug)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        except RuntimeError as e:
            # RuntimeError is common for CUDA OOM etc.
            raise VectorisationError(
                "Model forward pass failed (possible OOM/device issue).",
                context={
                    "vectoriser": "huggingface",
                    "model": self.model_name,
                    "n_texts": len(texts),
                    "device": str(self.device),
                },
            ) from e
        except Exception as e:
            raise VectorisationError(
                "Model forward pass failed.",
                context={"vectoriser": "huggingface", "model": self.model_name, "n_texts": len(texts)},
            ) from e

        # Pooling / output parsing
        try:
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts

            embeddings = mean_pooled.cpu().numpy()
        except Exception as e:
            raise VectorisationError(
                "Failed to compute embeddings from model outputs.",
                context={"vectoriser": "huggingface", "model": self.model_name},
            ) from e

        return embeddings
