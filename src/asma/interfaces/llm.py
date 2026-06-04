from abc import ABC, abstractmethod
from typing import Callable, Optional

class LLMProvider(ABC):
    @abstractmethod
    def respond(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None, 
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        """
        Sends a prompt (optionally with system instruction) to the LLM
        and returns the raw string content response.
        
        Args:
            prompt: The main user/input prompt.
            system_instruction: System prompt to set behavior.
            progress_callback: A callback function accepting a float progress value (0 to 1).
            stream: Whether to stream the response output live to console.
            **kwargs: Extra model-specific generation parameters (e.g., temperature, max_tokens).
        """
        pass

    def respond_chat(
        self,
        chat_history: list,
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        """
        Sends a multi-turn chat history (list of role/content dicts) to the LLM
        and returns the raw string content response of the final turn.
        
        Args:
            chat_history: List of dictionaries like [{"role": "system"/"user"/"assistant", "content": "..."}].
            progress_callback: Callback function for progress updates.
            stream: Whether to stream fragments/output.
        """
        raise NotImplementedError("Conversational chat is not supported by this provider.")
