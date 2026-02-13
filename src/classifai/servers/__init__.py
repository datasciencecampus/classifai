"""FastAPI server as a package."""

from .main import get_router, run_server

__all__ = ["get_router", "run_server"]
