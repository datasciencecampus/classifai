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
        hooks (dict): A dictionary of user-defined hooks for preprocessing and postprocessing.
    """

    def __init__(self, model_name, device=None, model_revision="main", hooks=None):
        """Initializes the HuggingfaceVectoriser with the specified model name and device.

        Args:
            model_name (str): The name of the Huggingface model to use.
            device (torch.device, optional): The device to use for computation. Defaults to GPU if available, otherwise CPU.
            model_revision (str, optional): The specific model revision to use. Defaults to "main".
            hooks (dict, optional): A dictionary of user-defined hooks for preprocessing and postprocessing. Defaults to None.
        """
        check_deps(["transformers", "torch"], extra="huggingface")
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)  # nosec: B615
        self.model = AutoModel.from_pretrained(model_name, revision=model_revision)  # nosec: B615

        # Use GPU if available and not overridden
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        self.hooks = {} if hooks is None else hooks

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

        if type(texts) is str:
            texts = [texts]

        if type(texts) is not list:
            raise TypeError("Input texts must be a string or a list of strings.")

        # Check if there is a user defined preprocess hook for the HFVectoriser transform method
        if "transform_preprocess" in self.hooks:
            # pass the args to the preprocessing function as a dictionary
            hook_output = self.hooks["transform_preprocess"]({"texts": texts})

            # Unpack the dictionary back into the argument variables
            texts = hook_output.get("texts", texts)

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

        # Check if there is a user defined postprocess hook for the HFVectoriser transform method
        if "transform_postprocess" in self.hooks:
            # pass the args to the postprocessing function as a dictionary
            hook_output = self.hooks["transform_postprocess"]({"embeddings": embeddings})

            # Unpack the dictionary back into the argument variables
            embeddings = hook_output.get("embeddings", embeddings)

        return embeddings
