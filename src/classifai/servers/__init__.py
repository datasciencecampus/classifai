"""REST API integration for ClassifAI vector stores.

This module provides functions for exposing one or more VectorStore
instances as a FastAPI service.

The generated API exposes the VectorStore `embed`, `search`, and
`reverse_search` methods as REST endpoints, allowing clients to perform
embedding and search operations over HTTP.

Full endpoint and Pydantic model documentation is available through the
automatically generated FastAPI Swagger UI at `/docs`.

To explore the API without providing your own data, first create a demo
VectorStore using `/DEMO/general_workflow_demo.ipynb`, then start a demo
server with `/DEMO/general_workflow_serve.py`.
"""

from .main import get_router, get_server, make_endpoints, run_server

__all__ = ["get_router", "get_server", "make_endpoints", "run_server"]
