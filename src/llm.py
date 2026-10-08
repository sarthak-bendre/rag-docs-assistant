"""LLM factory - swap between a local Ollama model and the Groq free-tier API."""
from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # 8K context fits 10 passages + answer; Ollama's larger default can exhaust a 6 GB GPU
        return ChatOllama(model=model, temperature=temperature, base_url=settings.ollama_base_url, num_ctx=8192)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)  # reads GROQ_API_KEY
    raise ValueError(f"Unknown LLM provider {provider!r} (use 'ollama' or 'groq')")
