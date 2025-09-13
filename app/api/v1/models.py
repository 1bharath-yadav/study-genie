from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.llm.providers import get_all_providers

router = APIRouter(prefix="/models", tags=["models"])

def _get_models_by_type(model_type: str) -> list[dict]:
    providers = get_all_providers()
    models = []
    for provider_name, provider_data in providers.items():
        for model_name in provider_data.get("models", []):
            # Assume all are chat for now, since embedding is separate
            if model_type == "chat":
                models.append({
                    "id": f"{provider_name}-{model_name}",
                    "provider_id": provider_name,
                    "model_name": model_name,
                    "display_name": model_name,
                    "model_type": "chat",
                    "is_active": True
                })
            elif model_type == "embedding" and provider_name == "google":  # example
                models.append({
                    "id": f"{provider_name}-embedding-001",
                    "provider_id": provider_name,
                    "model_name": "embedding-001",
                    "display_name": "Google Embeddings",
                    "model_type": "embedding",
                    "is_active": True
                })
    return models

@router.get("/available/chat")
async def get_available_chat_models(current_user: dict = Depends(get_current_user)):
    return _get_models_by_type("chat")

@router.get("/available/embedding")
async def get_available_embedding_models(current_user: dict = Depends(get_current_user)):
    return _get_models_by_type("embedding")