"""
LLM service integration with automatic API key retrieval
"""
from typing import Optional, Dict, Any
import logging

from app.core.auth_service import get_auth_service

logger = logging.getLogger(__name__)


class LLMKeyService:
    """Service for managing API keys for LLM services"""

    def __init__(self):
        self.auth_service = get_auth_service()

    async def get_api_key_for_user(self, user_id: str, service: str = "gemini") -> Optional[str]:
        """
        Get API key for a specific user and service
        This is the main method other services should use to get API keys
        """
        try:
            # Get the user's API key
            api_key = await self.auth_service.get_user_api_key_for_llm(user_id)

            if api_key:
                logger.info(
                    f"Retrieved API key for user {user_id} service {service}")
                return api_key
            else:
                logger.warning(
                    f"No API key found for user {user_id} service {service}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving API key for user {user_id}: {e}")
            return None

    async def validate_user_has_api_key(self, user_id: str) -> bool:
        """Check if user has a valid API key configured"""
        try:
            api_key = await self.get_api_key_for_user(user_id)
            return api_key is not None
        except Exception as e:
            logger.error(f"Error validating API key for user {user_id}: {e}")
            return False

    def create_llm_config_for_user(self, user_id: str, api_key: str, service: str = "gemini") -> Dict[str, Any]:
        """
        Create LLM configuration object for a specific user
        This can be used to configure LLM clients
        """
        if service.lower() == "gemini":
            return {
                "service": "gemini",
                "api_key": api_key,
                "model": "gemini-pro",
                "user_id": user_id
            }
        elif service.lower() == "openai":
            return {
                "service": "openai",
                "api_key": api_key,
                "model": "gpt-3.5-turbo",
                "user_id": user_id
            }
        elif service.lower() == "anthropic":
            return {
                "service": "anthropic",
                "api_key": api_key,
                "model": "claude-3-sonnet",
                "user_id": user_id
            }
        else:
            return {
                "service": service,
                "api_key": api_key,
                "user_id": user_id
            }


# Global LLM key service instance
_llm_key_service: Optional[LLMKeyService] = None


def get_llm_key_service() -> LLMKeyService:
    """Get or create global LLM key service instance"""
    global _llm_key_service
    if _llm_key_service is None:
        _llm_key_service = LLMKeyService()
    return _llm_key_service
