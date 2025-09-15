"""Compatibility shim for older imports that referenced `app.llm.provider`.

This module re-exports a small set of helper functions from
`app.llm.providers` so older code that imports `app.llm.provider` continues to
work while we keep the canonical implementation in `providers.py`.
"""
from app.llm.providers import (
    get_user_model_preferences,
    get_provider_by_id,
    get_all_providers,
    get_provider_list,
    get_models_for_provider,
)

__all__ = [
    "get_user_model_preferences",
    "get_provider_by_id",
    "get_all_providers",
    "get_provider_list",
    "get_models_for_provider",
]
