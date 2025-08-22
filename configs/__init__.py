"""Configuration package for study-genie.

Exports commonly used configuration values so other modules can simply do:
    from configs import API_HOST, API_PORT

This module reads from environment variables and provides sensible defaults.
"""
from os import environ

# Defaults
DEFAULT_API_HOST = environ.get("FASTAPI_HOST", "0.0.0.0")
DEFAULT_API_PORT = int(environ.get("FASTAPI_PORT", "8000"))

API_HOST = environ.get("API_HOST", DEFAULT_API_HOST)
try:
    API_PORT = int(environ.get("API_PORT", DEFAULT_API_PORT))
except (TypeError, ValueError):
    API_PORT = DEFAULT_API_PORT

__all__ = ["API_HOST", "API_PORT"]
