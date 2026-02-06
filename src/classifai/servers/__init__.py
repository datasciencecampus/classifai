"""FastAPI server as a package."""

from .main import ClassifAIServer, start_api

__all__ = [
    "ClassifAIServer",
    "start_api",
]
