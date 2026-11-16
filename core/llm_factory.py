"""
core/llm_factory.py - Central Factory for Initializing LLM Models.
Supports:
- Google Gemini (via ChatGoogleGenerativeAI)
- OpenRouter / OpenAI-compatible models (via ChatOpenAI)
"""

from typing import Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import settings

def get_chat_model(
    provider: Literal['gemini','openrouter'] = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
    ) -> BaseChatModel:

    """
        Returns a unified LangChain BaseChatModel instance.
        Args:
            provider: 'gemini' or 'openrouter'
            model_name: Specific model ID (defaults to config settings if None)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Optional output token limit
            **kwargs: Extra parameters passed to the model provider
    """

    if provider == "gemini":
        target_model = model_name or settings.fast_model
        api_key = settings.gemini_api_key

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")

        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=45.0,
            **kwargs
        )
    elif provider == "openrouter":
        target_model = model_name or settings.openrouter_default_model
        api_key = settings.openrouter_api_key

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")

        # OpenRouter exposes an OpenAI-compatible endpoint
        return ChatOpenAI(
            model=target_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": "https://github.com/agentic-desktop-assistant",
                "X-Title": "Agentic Desktop Assistant",
            },
            **kwargs
        )


    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Supported: 'gemini', 'openrouter'")


def extract_text(response) -> str:
    """Extracts a clean text string from any LangChain AIMessage response."""
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
        return "".join(texts)
    return str(content)
