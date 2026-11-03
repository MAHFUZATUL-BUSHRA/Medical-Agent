"""LLM factory: one place that knows about OpenAI / Groq / Gemini."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from . import config


def make_chat_model() -> ChatOpenAI:
    """LangChain chat model (used by the SQL agents and the LangChain router)."""
    config.require_llm_key()
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,
        timeout=60,
        max_retries=1,
    )


def make_sdk_model():
    """Model for the OpenAI Agents SDK router."""
    config.require_llm_key()
    if config.LLM_PROVIDER == "openai":
        return config.LLM_MODEL
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    from openai import AsyncOpenAI

    set_tracing_disabled(True)  # tracing uploads to OpenAI and needs an OpenAI key
    client = AsyncOpenAI(
        api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, timeout=60, max_retries=1
    )
    return OpenAIChatCompletionsModel(model=config.LLM_MODEL, openai_client=client)
