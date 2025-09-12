"""
Pure functional LLM providers using PydanticAI
"""
from pydantic_ai import Agent
from typing import Optional
from .types import LLMRequest, LLMResponse, StudyContent, QuestionAnswer


# Provider Configuration Functions
def create_openai_model(api_key: str, model_name: str = "gpt-4") -> str:
    """Create OpenAI model configuration string."""
    return f"openai:{model_name}"


def create_google_model(api_key: str, model_name: str = "gemini-1.5-pro") -> str:
    """Create Google Gemini model configuration string.""" 
    return f"gemini:{model_name}"


def create_anthropic_model(api_key: str, model_name: str = "claude-3-5-sonnet") -> str:
    """Create Anthropic model configuration string."""
    return f"anthropic:{model_name}"


# Study-focused Agent Creation Functions
def create_study_agent(model_config: str, system_prompt: Optional[str] = None) -> Agent[None, str]:
    """Create a basic study assistant agent."""
    default_prompt = """You are a helpful study assistant. Help students learn effectively by:
    - Explaining concepts clearly
    - Providing structured summaries
    - Creating study plans
    - Answering questions with educational value"""
    
    return Agent(
        model_config,
        system_prompt=system_prompt or default_prompt
    )


def create_content_agent(model_config: str) -> Agent[None, StudyContent]:
    """Create an agent that returns structured study content."""
    system_prompt = """You are a content structuring assistant. Transform any educational material into structured study content with:
    - Clear title
    - Concise summary 
    - Key learning points
    - Important concepts
    - Appropriate difficulty level"""
    
    return Agent(
        model_config,
        output_type=StudyContent,
        system_prompt=system_prompt
    )


def create_qa_agent(model_config: str) -> Agent[None, QuestionAnswer]:
    """Create an agent for generating Q&A pairs."""
    system_prompt = """You are a question and answer generator. Create educational Q&A pairs that:
    - Ask meaningful questions about the topic
    - Provide clear, accurate answers
    - Include helpful explanations
    - Identify the specific topic being covered"""
    
    return Agent(
        model_config,
        output_type=QuestionAnswer,
        system_prompt=system_prompt
    )


# Core LLM Functions
async def generate_text(request: LLMRequest, api_key: str) -> LLMResponse:
    """Generate text using the appropriate provider."""
    # Determine provider and create model config
    provider = request.model_id.split('-')[0].lower() if '-' in request.model_id else 'gpt'
    
    if provider in ['gpt', 'openai']:
        model_config = create_openai_model(api_key, request.model_id)
    elif provider in ['gemini', 'google']:
        model_config = create_google_model(api_key, request.model_id)
    elif provider in ['claude', 'anthropic']:
        model_config = create_anthropic_model(api_key, request.model_id)
    else:
        model_config = create_openai_model(api_key, "gpt-4")  # Default fallback
    
    # Create agent and run
    agent = create_study_agent(model_config, request.system_prompt)
    
    try:
        result = await agent.run(request.prompt)
        
        return LLMResponse(
            content=str(result.output),
            model_id=request.model_id,
            provider=provider,
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(str(result.output).split()),
                "total_tokens": len(request.prompt.split()) + len(str(result.output).split())
            },
            finish_reason="stop"
        )
    except Exception as e:
        return LLMResponse(
            content=f"Error: {str(e)}",
            model_id=request.model_id,
            provider=provider,
            usage=None,
            finish_reason="error"
        )


async def generate_structured_content(prompt: str, model_id: str, api_key: str) -> StudyContent:
    """Generate structured study content."""
    provider = model_id.split('-')[0].lower() if '-' in model_id else 'gpt'
    
    if provider in ['gpt', 'openai']:
        model_config = create_openai_model(api_key, model_id)
    elif provider in ['gemini', 'google']:
        model_config = create_google_model(api_key, model_id)
    elif provider in ['claude', 'anthropic']:
        model_config = create_anthropic_model(api_key, model_id)
    else:
        model_config = create_openai_model(api_key, "gpt-4")
        
    agent = create_content_agent(model_config)
    result = await agent.run(prompt)
    return result.output


async def generate_qa_pair(prompt: str, model_id: str, api_key: str) -> QuestionAnswer:
    """Generate Q&A pairs for study."""
    provider = model_id.split('-')[0].lower() if '-' in model_id else 'gpt'
    
    if provider in ['gpt', 'openai']:
        model_config = create_openai_model(api_key, model_id)
    elif provider in ['gemini', 'google']:
        model_config = create_google_model(api_key, model_id)
    elif provider in ['claude', 'anthropic']:
        model_config = create_anthropic_model(api_key, model_id)
    else:
        model_config = create_openai_model(api_key, "gpt-4")
        
    agent = create_qa_agent(model_config)
    result = await agent.run(prompt)
    return result.output


# Provider capability mapping
PROVIDER_CAPABILITIES = {
    "openai": {
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "capabilities": ["text_generation", "function_calling"]
    },
    "google": {
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "capabilities": ["text_generation", "vision", "function_calling"]
    },
    "anthropic": {
        "models": ["claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus"],
        "capabilities": ["text_generation", "function_calling"]
    }
}


def get_supported_models(provider: str) -> list[str]:
    """Get supported models for a provider."""
    return PROVIDER_CAPABILITIES.get(provider, {}).get("models", [])


def get_provider_capabilities(provider: str) -> list[str]:
    """Get capabilities for a provider."""
    return PROVIDER_CAPABILITIES.get(provider, {}).get("capabilities", [])


def get_all_providers() -> dict:
    """Get all available providers and their capabilities."""
    return PROVIDER_CAPABILITIES
