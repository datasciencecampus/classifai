"""Config class for apps."""

import logging
import os

from dotenv import dotenv_values
from google.cloud import logging as cloud_logging
from google.cloud.secretmanager import SecretManagerServiceClient


class Config:
    """Simple configuration class that checks environment variables and secrets."""

    def __init__(self, service_name="API"):
        env = dotenv_values(".env")
        # Environment detection
        self.service_name = service_name
        self.env_type = os.getenv("ENV_TYPE") or env.get("ENV_TYPE") or "dev"
        self.api_type = os.getenv("API_TYPE") or env.get("API_TYPE") or "live"
        self.is_production = False
        self.project_id = os.getenv("PROJECT_ID") or env.get("PROJECT_ID")
        self.embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME") or env.get("EMBEDDINGS_MODEL_NAME") or "text-embedding-004"
        self.embeddings_model_task = os.getenv("EMBEDDINGS_MODEL_TASK") or env.get("EMBEDDINGS_MODEL__TASK") or "CLASSIFICATION"
        self.db_dir = "data/db" if self.api_type == "local" else "/tmp/"

        if service_name == "API":
            if self.api_type != "local":
                self.is_production = True

        # Core configuration
        if service_name == "API":
            self.api_url = None
            self.oauth_client_id = None
            self.bucket_name = (
                os.getenv("BUCKET_NAME")
                or env.get("BUCKET_NAME")
                or self._get_secret("APP_DATA_BUCKET")
            )
            self.embedding_api_key = (
                os.getenv("EMBEDDING_API_KEY")
                or env.get("EMBEDDING_API_KEY")
                or self._get_secret("GOOGLE_API_KEY")
            )

    def _get_secret(self, secret_id, default=None):
        """Get a secret from Google Secret Manager."""
        if not self.project_id:
            return default

        try:
            client = SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(name=name)
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"Failed to retrieve secret {secret_id}: {e}")
            return default

    def setup_logging(self):
        """Set up logging appropriate for the environment."""
        log_level = logging.DEBUG if not self.is_production else logging.INFO

        if not self.is_production:
            # Local development logging
            logging.basicConfig(encoding="utf-8", level=log_level)
        else:
            # Production logging to Google Cloud
            try:
                # Initialize Google Cloud Logging client
                client = cloud_logging.Client(project=self.project_id)
                client.setup_logging(log_level=log_level)
            except Exception as e:
                # Fallback to basic logging
                print(f"Error setting up Cloud Logging: {e}")
                logging.basicConfig(encoding="utf-8", level=log_level)

    def validate(self):
        """Validate that all required config is present."""
        # Common required configs
        required = ["project_id", "api_type"]

        # Add environment-specific required configs
        if self.service_name == "API":
            required.extend(["bucket_name", "embedding_api_key"])

        missing = []
        for item in required:
            value = getattr(self, item, None)
            if value is None or (isinstance(value, list) and len(value) == 0):
                missing.append(item)

        if missing:
            logging.error(
                f"Missing required configuration: {', '.join(missing)}"
            )
            return False

        return True

def get_secret(secret_name: str, project_id: str):
    """Access GCP Secret Manager secret value.

    Parameters
    ----------
    secret_name : string
        Name of secret to access.
    project_id : string
        GCP project name.

    Returns
    -------
    secrets : str|json
        Secret value.
    """

    env_var_project_id = os.getenv("PROJECT_ID")
    if env_var_project_id:
        project_id = env_var_project_id
    path = f"projects/{project_id}/secrets"
    name = "/".join((path, secret_name, "versions", "latest"))
    client = SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": name})
    secret_value = response.payload.data.decode("UTF-8")

    return secret_value
