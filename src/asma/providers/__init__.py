from asma.providers.llm_gemini import GeminiProvider, AVAILABLE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL
from asma.providers.llm_lmstudio import LMStudioProvider
from asma.providers.resolver_crossref import CrossrefResolver
from asma.providers.fetcher_pmc import PmcFetcher


def get_llm_provider(settings):
    """
    Factory function returning an instantiated LLMProvider
    based on settings instance or dictionary (Gemini Cloud API or LM Studio Local).
    """
    provider_type = (getattr(settings, "llm_provider", "gemini") if not isinstance(settings, dict) else settings.get("llm_provider", "gemini")) or "gemini"
    provider_type = str(provider_type).lower()

    if provider_type == "gemini":
        api_key = getattr(settings, "gemini_api_key", "") if not isinstance(settings, dict) else settings.get("gemini_api_key", "")
        model_name = (getattr(settings, "gemini_model_name", DEFAULT_GEMINI_MODEL) if not isinstance(settings, dict) else settings.get("gemini_model_name", DEFAULT_GEMINI_MODEL)) or DEFAULT_GEMINI_MODEL
        enable_thinking = getattr(settings, "enable_thinking", True) if not isinstance(settings, dict) else settings.get("enable_thinking", True)
        return GeminiProvider(
            api_key=api_key,
            model_name=model_name,
            enable_thinking=enable_thinking
        )
    else:
        model_name = (getattr(settings, "llm_model_name", "google/gemma-4-e2b-qat") if not isinstance(settings, dict) else settings.get("llm_model_name", "google/gemma-4-e2b-qat")) or "google/gemma-4-e2b-qat"
        base_url = (getattr(settings, "llm_base_url", "http://localhost:1234") if not isinstance(settings, dict) else settings.get("llm_base_url", "http://localhost:1234")) or "http://localhost:1234"
        return LMStudioProvider(
            model_name=model_name,
            base_url=base_url
        )


__all__ = [
    "GeminiProvider",
    "AVAILABLE_GEMINI_MODELS",
    "DEFAULT_GEMINI_MODEL",
    "LMStudioProvider",
    "CrossrefResolver",
    "PmcFetcher",
    "get_llm_provider"
]
