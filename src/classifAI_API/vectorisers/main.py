from google import genai
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)



class Gcp_Vectoriser():
    """
        A class for embedding text using Google Cloud Platform's GenAI API.
        Attributes:
            model_name (str): The name of the embedding model to use.
            vectoriser (genai.Client): The GenAI client instance for embedding text.
    """


    def __init__(self, project_id, location='europe-west2', model_name="text-embedding-004"): 
        """
            Initializes the Gcp_Vectoriser with the specified project ID, location, and model name.
            Args:
                project_id (str): The Google Cloud project ID.
                location (str, optional): The location of the GenAI API. Defaults to 'europe-west2'.
                model_name (str, optional): The name of the embedding model. Defaults to "text-embedding-004".
            Raises:
                RuntimeError: If the GenAI client fails to initialize.
        """

        self.model_name = model_name

        try:
            self.vectoriser = genai.Client(
                vertexai=True, 
                project="classifai-sandbox",
                location=location,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCP Vectoriser through ganai.Client API: {e}")



    def transform(self, texts):
        """
            Transforms input text(s) into embeddings using the GenAI API.
            Args:
                texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.
            Returns:
                numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.
            Raises:
                TypeError: If the input is not a string or a list of strings.
        """

        if type(texts) is str:
            texts = [texts]
        
        if type(texts) is not list:
            raise TypeError("Input must be a string or a list of strings.")

        
        # The Vertex AI call to  embed content
        embeddings = self.vectoriser.models.embed_content(
            model=self.model_name,
            contents=texts,
        )

        # Extract embeddings from the response object
        #embeddings = [embedding[0] for embedding in embeddings]
        result = np.array([res.values for res in embeddings.embeddings])
            
        return result

    

#general wrapper class for Huggingface Transformers models, may not work for all models
class Huggingface_Vectoriser:
    """
        A general wrapper class for Huggingface Transformers models to generate text embeddings.
        Attributes:
            model_name (str): The name of the Huggingface model to use.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the specified model.
            model (transformers.PreTrainedModel): The Huggingface model instance.
            device (torch.device): The device (CPU or GPU) on which the model is loaded.
    """


    def __init__(self, model_name, device=None):
        """
            Initializes the Huggingface_Vectoriser with the specified model name and device.
            Args:
                model_name (str): The name of the Huggingface model to use.
                device (torch.device, optional): The device to use for computation. Defaults to GPU if available, otherwise CPU.
        """

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Use GPU if available and not overridden
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()


    def transform(self, texts):
        """
            Transforms input text(s) into embeddings using the Huggingface model.
            Args:
                texts (str or list of str): The input text(s) to embed. Can be a single string or a list of strings.
            Returns:
                numpy.ndarray: A 2D array of embeddings, where each row corresponds to an input text.
            Raises:
                TypeError: If the input is not a string or a list of strings.
        """

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list):
            raise TypeError("Input must be a string or a list of strings.")

        # Tokenise input texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        #Use mean pooling over the token embeddings
        token_embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        attention_mask = inputs['attention_mask']

        # Perform mean pooling manually
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts  # shape: (batch_size, hidden_size)

        # Convert to numpy array
        embeddings = mean_pooled.cpu().numpy()

        return embeddings