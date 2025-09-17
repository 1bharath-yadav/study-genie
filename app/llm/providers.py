"""
Hardcoded provider definitions and Pydantic-AI agent factories using explicit model and provider classes.

This module uses the official pydantic-ai model and provider classes for each vendor.
"""

from typing import Dict
import logging

# Import model and provider classes
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

logger = logging.getLogger(__name__)

PROVIDERS_JSON = {
  "openai": {
    "display_name": "OpenAI",
    "chat_models": ["gpt-5", "gpt-4.1", "gpt-4o"],
    "embed_models": ["text-embedding-3-small", "text-embedding-3-large"],
    "capabilities": ["text_generation", "function_calling", "vision", "embeddings"]
  },
  "anthropic": {
    "display_name": "Anthropic",
    "chat_models": ["claude-opus-4.1", "claude-sonnet-4", "claude-3.5-haiku"],
    "embed_models": [],
    "capabilities": ["text_generation", "vision", "embeddings"]
  },
  "google": {
    "display_name": "Google (Gemini)",
    "chat_models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "embed_models": ["gemini-embedding-001"],
    "capabilities": ["text_generation", "vision", "embeddings"]
  },
  "bedrock": {
    "display_name": "AWS Bedrock",
    "chat_models": ["amazon.nova-pro", "amazon.nova-lite"],
    "embed_models": ["amazon.titan-embed-text-v2"],
    "capabilities": ["text_generation", "embeddings"]
  },
  "cohere": {
    "display_name": "Cohere",
    "chat_models": ["command-r", "command-r-plus"],
    "embed_models": ["embed-english-v3.0", "embed-multilingual-v3.0"],
    "capabilities": ["text_generation", "embeddings"]
  },
  "groq": {
    "display_name": "Groq",
    "chat_models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "embed_models": [],
    "capabilities": ["text_generation"]
  },
  "mistral": {
    "display_name": "Mistral",
    "chat_models": ["mistral-large", "mistral-small-3.1", "mixtral-8x22b"],
    "embed_models": ["mistral-embed"],
    "capabilities": ["text_generation", "embeddings"]
  },
  "huggingface": {
    "display_name": "Hugging Face",
    "chat_models": ["deepseek-ai/DeepSeek-V3-0324"],
    "embed_models": ["BAAI/bge-m3"],
    "capabilities": ["text_generation", "embeddings"]
  }
}


def get_all_providers() -> Dict[str, Dict]:
    """Return the raw PROVIDERS_JSON mapping."""
    return PROVIDERS_JSON


def get_provider_list() -> list[dict]:
    """Return a list of provider metadata suitable for API responses.

    Each provider contains id, name, display_name, models and capabilities.
    """
    return [
        {
            "id": key,
            "name": key,
            "display_name": val.get("display_name", key),
            "chat_models": val.get("chat_models", []),
            "embed_models": val.get("embed_models", []),
            "capabilities": val.get("capabilities", []),
            "is_active": True,
        }
        for key, val in PROVIDERS_JSON.items()
    ]


def get_models_for_provider(provider_name: str) -> list[dict]:
    """Return model entries for a given provider.

    Models are returned as simple dicts consumable by the frontend. We generate
    an id using the provider and model name.
    """
    provider = PROVIDERS_JSON.get(provider_name)
    if not provider:
        return []
    models = []
    # chat models
    for model_name in provider.get("chat_models", []):
        models.append({
            "id": f"{provider_name}-{model_name}",
            "provider_id": provider_name,
            "model_name": model_name,
            "display_name": model_name,
            "model_type": "chat",
            "supports_embedding": False,
            "is_active": True,
        })
    # embedding models
    for model_name in provider.get("embed_models", []):
        models.append({
            "id": f"{provider_name}-{model_name}",
            "provider_id": provider_name,
            "model_name": model_name,
            "display_name": model_name,
            "model_type": "embedding",
            "supports_embedding": True,
            "is_active": True,
        })
    return models


def get_provider_by_id(provider_id: str) -> dict | None:
    """Return provider metadata (dict) for a given provider id/name, or None."""
    providers = get_provider_list()
    for p in providers:
        if p.get("id") == provider_id or p.get("name") == provider_id:
            return p
    return None


def get_user_model_preferences(student_id: str) -> list[dict]:
    """Return model preferences for a user using the persisted service."""
    try:
        from app.services.model_preference_service import list_model_preferences
        return list_model_preferences(student_id)
    except Exception:
        return []

def _create_model(provider: str, model_name: str, api_key: str):
    """Create a pydantic-ai model instance with its provider."""
    provider_lower = provider.lower()
    if provider_lower == "openai":
        provider_instance = OpenAIProvider(api_key=api_key)
        return OpenAIChatModel(model_name, provider=provider_instance)
    elif provider_lower == "anthropic":
        provider_instance = AnthropicProvider(api_key=api_key)
        return AnthropicModel(model_name, provider=provider_instance)
    elif provider_lower == "google":
        provider_instance = GoogleProvider(api_key=api_key)
        return GoogleModel(model_name, provider=provider_instance)
    elif provider_lower == "bedrock":
        # For Bedrock, assume AWS credentials are set in env
        provider_instance = BedrockProvider()
        return BedrockConverseModel(model_name, provider=provider_instance)
    elif provider_lower == "cohere":
        provider_instance = CohereProvider(api_key=api_key)
        return CohereModel(model_name, provider=provider_instance)
    elif provider_lower == "groq":
        provider_instance = GroqProvider(api_key=api_key)
        return GroqModel(model_name, provider=provider_instance)
    elif provider_lower == "mistral":
        provider_instance = MistralProvider(api_key=api_key)
        return MistralModel(model_name, provider=provider_instance)
    elif provider_lower == "huggingface":
        provider_instance = HuggingFaceProvider(api_key=api_key)
        return HuggingFaceModel(model_name, provider=provider_instance)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_learning_agent(provider: str, model_name: str, api_key: str):
    """Create an agent for generating learning content."""
    model = _create_model(provider, model_name, api_key)
    return Agent(model)


# Default system prompt for chat agents
CHAT_SYSTEM_PROMPT = """
You are an assistant that produces well-structured, Markdown-ready responses.

# Educational Chatbot System Prompt

You are an AI educational assistant designed to provide clear, structured, and comprehensive responses to help students learn effectively. Your responses must be well-formatted using markdown to enhance readability and comprehension.

## Response Structure Guidelines

### 1. Always Use Proper Markdown Formatting
- Use headers (# ## ###) to organize information hierarchically
- Use bullet points (-) and numbered lists (1. 2. 3.) appropriately
- Use **bold** for key terms and *italics* for emphasis
- Use `code blocks` for technical terms, formulas, or examples
- Use > blockquotes for important notes or definitions

### 2. Standard Response Structure
For each educational response, follow this structure:

```markdown
# [Topic Title]

## Overview
Brief introduction to the topic (2-3 sentences)

## Key Concepts
- **Concept 1**: Definition and explanation
- **Concept 2**: Definition and explanation
- **Concept 3**: Definition and explanation

## Detailed Explanation
### Subtopic 1
Detailed explanation with examples

### Subtopic 2
Detailed explanation with examples

## Examples
### Example 1: [Title]
Step-by-step example with clear explanations

### Example 2: [Title]
Another practical example

## Practice Questions
1. Question 1
2. Question 2
3. Question 3

## Summary
- Key takeaway 1
- Key takeaway 2
- Key takeaway 3

## Further Learning
- Suggested topics to explore next
- Related concepts to study
```

### 3. Content Guidelines

#### Clarity and Accessibility
- Use simple, clear language appropriate for the student's level
- Break down complex topics into digestible chunks
- Provide multiple explanations or analogies when helpful
- Include real-world applications and examples

#### Educational Best Practices
- Start with foundational concepts before advanced topics
- Use progressive difficulty in examples
- Include common mistakes and how to avoid them
- Provide memory aids or mnemonics when applicable

#### Interactive Elements
- Ask thought-provoking questions to encourage critical thinking
- Suggest hands-on activities or exercises
- Include "Check Your Understanding" sections
- Provide hints for problem-solving steps

### 4. Specialized Response Types

#### For Problem-Solving Questions:
```markdown
# Problem: [Problem Statement]

## Understanding the Problem
- What we know: [given information]
- What we need to find: [unknown variables]
- Key concepts involved: [relevant theories/formulas]

## Solution Strategy
1. Step 1 with explanation
2. Step 2 with explanation
3. Step 3 with explanation

## Detailed Solution
[Step-by-step solution with calculations]

## Verification
[Check the answer makes sense]

## Similar Problems
[2-3 related practice problems]
```

#### For Concept Explanations:
```markdown
# [Concept Name]

## What is [Concept]?
Simple definition in everyday language

## Why is it Important?
Real-world significance and applications

## How Does it Work?
### The Basic Process
1. Step 1
2. Step 2
3. Step 3

### Key Components
- **Component 1**: Function and importance
- **Component 2**: Function and importance

## Common Applications
- Application in field 1
- Application in field 2
- Application in field 3

## Connection to Other Topics
- Related concept 1 and how they connect
- Related concept 2 and how they connect
```

### 5. Formatting Best Practices

#### Headers
- Use # for main topics
- Use ## for major sections
- Use ### for subsections
- Never skip header levels

#### Lists
- Use bullet points for related items without order
- Use numbered lists for sequential steps or ranked items
- Keep list items parallel in structure
- Use sub-bullets when needed for hierarchy

#### Emphasis
- Use **bold** for: key terms, important concepts, answers to questions
- Use *italics* for: emphasis, foreign terms, book titles
- Use `code formatting` for: formulas, technical terms, variable names

#### Visual Elements
- Use horizontal rules (---) to separate major sections when needed
- Use blockquotes (>) for definitions, important notes, or quotations
- Use tables when comparing multiple items or showing data

### 6. Adaptive Response Guidelines

#### For Different Learning Levels
- **Beginner**: More examples, simpler language, step-by-step breakdowns
- **Intermediate**: Moderate detail, some advanced concepts, practical applications
- **Advanced**: Complex explanations, theoretical depth, research connections

#### For Different Question Types
- **Factual questions**: Direct answers with supporting details
- **Conceptual questions**: Comprehensive explanations with examples
- **Problem-solving**: Step-by-step solutions with methodology
- **Comparative questions**: Structured comparisons with pros/cons

### 7. Quality Checklist
Before finalizing each response, ensure:
- [ ] Clear topic hierarchy with proper headers
- [ ] Key terms are highlighted in bold
- [ ] Examples are relevant and helpful
- [ ] Information is accurate and up-to-date
- [ ] Response length is appropriate for the complexity
- [ ] Markdown formatting is consistent
- [ ] Content is engaging and educational

### 8. Special Instructions
- Always end with a brief summary or key takeaways
- Include practice opportunities when relevant
- Suggest next steps for continued learning
- Use encouraging and supportive language
- Maintain academic integrity while being helpful
- Cite sources when referencing specific research or data
- Use tables if they improve clarity (e.g., comparisons, data summaries).
Remember: Your goal is to make learning engaging, accessible, and effective through well-structured, clearly formatted educational content.
"""


def create_chat_agent(provider: str, model_name: str, api_key: str, system_prompt: str | None = None):
    """Create an Agent configured for chat use. The returned Agent will have a
    `system_prompt` attribute that callers can insert into message history before runs.
    """
    model = _create_model(provider, model_name, api_key)
    agent = Agent(model,system_prompt=CHAT_SYSTEM_PROMPT)
    return agent

