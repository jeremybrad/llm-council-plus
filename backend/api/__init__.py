"""API routers package.

This package contains FastAPI routers for the various API endpoints.
Import routers here for easy inclusion in main.py.
"""

from .claims import router as claims_router

__all__ = ["claims_router"]
