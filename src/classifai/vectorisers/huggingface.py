"""A module that provides a wrapper for Huggingface Transformers models to generate text embeddings."""

from classifai._optional import check_deps

from .base import VectoriserBase


class HuggingFaceVectoriser(VectoriserBase):
    """A general wrapper class for Huggingface Transformers models to generate text embeddings.

    Attributes:
        model_name (str): The name of the Huggingface model to use.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the specified model.
        model (transformers.PreTrainedModel): The Huggingface model instance.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        tokenizer_kwargs (dict): Additional keyword arguments passed to the tokenizer.
        model_kwargs (dict): Additional keyword arguments passed to the model.
    """

    def __init__(
        self,
        model_name,
        device=None,
        model_revision="main",
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ):
        """Initializes the HuggingfaceVectoriser with the specified model name and device.

        Args:
            model_name (str): The name of the Huggingface model to use.
            device (torch.device, optional): The device to use for computation. Defaults to GPU if available, otherwise CPU.
            model_revision (str, optional): The specific model revision to use. Defaults to "main".
            tokenizer_kwargs (dict, optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Defaults to None.
        """
        check_deps(["transformers", "torch"], extra="huggingface")
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.model_name = model_name

        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        model_kwargs = dict(model_kwargs or {})

        # Ensure consistent behavior unless user overrides it
        tokenizer_kwargs.setdefault("trust_remote_code", False)
        model_kwargs.setdefault("trust_remote_code", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            **tokenizer_kwargs,
        )  # nosec: B615
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
            **model_kwargs,
        )  # nosec: B615

        # Use GPU if available and not overridden
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    def transform(self, texts):
        """Transforms input text(s) into embeddings using the Huggingface model.

        Args:
            texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.

        Raises:
            TypeError: If the input is not a string or a list of strings.
        """
        import torch  # type: ignore

        # If a single string is passed as arg to texts, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenise input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling over the token embeddings
        token_embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        attention_mask = inputs["attention_mask"]

        # Perform mean pooling manually
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts  # shape: (batch_size, hidden_size)

        # Convert to numpy array
        embeddings = mean_pooled.cpu().numpy()

        return embeddings
