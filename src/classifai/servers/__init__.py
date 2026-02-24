"""This module provides functionality for creating or extending a REST-API service
which allows a user to call the search methods of one or more VectorStore objects,
from an API endpoint.

These functions interact with the ClassifAI Indexer module's VectorStore objects,
such that their `embed`, `search` and `reverse_search` methods are exposed on
REST-API endpoints, via a FastAPI service.
"""

from .main import get_router, get_server, make_endpoints, run_server

__all__ = ["get_router", "get_server", "make_endpoints", "run_server"]
