import logging
from typing import Callable, Optional
from asma.interfaces.llm import LLMProvider

logger = logging.getLogger(__name__)

class LMStudioProvider(LLMProvider):
    """
    Client wrapper for LM Studio's Python SDK to perform inference.
    """
    def __init__(self, model_name: str = "google/gemma-4-e2b"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import lmstudio as lms
            try:
                lms.set_sync_api_timeout(3600.0)
                logger.info("Set LM Studio SDK message timeout to 3600.0 seconds (1 hour) to support large contexts.")
            except Exception as e:
                logger.warning(f"Could not set LM Studio sync API timeout: {e}")
            logger.info(f"Connecting to local LM Studio model: {self.model_name}...")
            try:
                client = lms.Client()
                self._model = client.llm.model(self.model_name)
            except Exception as e:
                logger.error(f"Failed to connect to LM Studio model '{self.model_name}': {e}")
                raise RuntimeError(f"Could not load LM Studio model '{self.model_name}'. Is LM Studio running? Error: {e}") from e
        return self._model

    def respond(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None, 
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        import lmstudio as lms
        import sys
        
        chat = lms.Chat()
        if system_instruction:
            chat.add_system_prompt(system_instruction)
        chat.add_user_message(prompt)
        
        # 1. Safety check: Token count vs Context length
        ignore_limit = kwargs.pop("ignore_context_limit", False)
        try:
            prompt_str = self.model.apply_prompt_template(chat)
            tokens = len(self.model.tokenize(prompt_str))
            context_len = self.model.get_context_length()
            
            logger.info(f"Prompt token count: {tokens} | Model context length: {context_len}")
            
            if tokens > context_len and not ignore_limit:
                raise ValueError(
                    f"Prompt contains {tokens} tokens, which exceeds the model's context window limit "
                    f"of {context_len} tokens. This will cause truncation or failure. "
                    "Shorten your document or pass `ignore_context_limit=True` to bypass this check."
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Could not compute context window limit safety check: {e}")

        # 2. Setup progress/streaming callbacks
        # Always configure prompt reading progress by default to keep user updated
        prompt_progress_fn = progress_callback
        if prompt_progress_fn is None and "on_prompt_processing_progress" not in kwargs:
            def default_prompt_progress(progress: float):
                percent = int(progress * 100)
                sys.stdout.write(f"\r[LM Studio] Reading prompt: {percent}%")
                sys.stdout.flush()
                if percent >= 100:
                    sys.stdout.write("\n[LM Studio] Generating response...\n")
                    sys.stdout.flush()
            kwargs["on_prompt_processing_progress"] = default_prompt_progress
        elif prompt_progress_fn is not None:
            kwargs["on_prompt_processing_progress"] = prompt_progress_fn

        # Configure live prediction token streaming only if stream=True
        if stream:
            if "on_prediction_fragment" not in kwargs:
                def default_prediction_fragment(frag):
                    sys.stdout.write(frag.content)
                    sys.stdout.flush()
                kwargs["on_prediction_fragment"] = default_prediction_fragment

        logger.info("Sending request to LM Studio...")
        try:
            result = self.model.respond(
                chat,
                **kwargs
            )
            # Ensure a clean newline after streaming output if streaming was active
            if stream:
                print()
            return result.content
        except Exception as e:
            logger.error(f"LM Studio response generation failed: {e}")
            self._model = None
            raise RuntimeError(f"LM Studio inference error: {e}") from e

    def respond_chat(
        self,
        chat_history: list,
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        import lmstudio as lms
        import sys
        
        chat = lms.Chat()
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                chat.add_system_prompt(content)
            elif role == "user":
                chat.add_user_message(content)
            elif role == "assistant":
                chat.add_assistant_response(content)

        # 1. Safety check: Token count vs Context length
        ignore_limit = kwargs.pop("ignore_context_limit", False)
        try:
            prompt_str = self.model.apply_prompt_template(chat)
            tokens = len(self.model.tokenize(prompt_str))
            context_len = self.model.get_context_length()
            
            logger.info(f"Chat history token count: {tokens} | Model context length: {context_len}")
            
            if tokens > context_len and not ignore_limit:
                raise ValueError(
                    f"Chat history contains {tokens} tokens, which exceeds the model's context window limit "
                    f"of {context_len} tokens. This will cause truncation or failure. "
                    "Shorten your document or pass `ignore_context_limit=True` to bypass this check."
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Could not compute context window limit safety check: {e}")

        # 2. Setup progress/streaming callbacks
        prompt_progress_fn = progress_callback
        if prompt_progress_fn is None and "on_prompt_processing_progress" not in kwargs:
            def default_prompt_progress(progress: float):
                percent = int(progress * 100)
                sys.stdout.write(f"\r[LM Studio] Reading chat prompt: {percent}%")
                sys.stdout.flush()
                if percent >= 100:
                    sys.stdout.write("\n[LM Studio] Generating response...\n")
                    sys.stdout.flush()
            kwargs["on_prompt_processing_progress"] = default_prompt_progress
        elif prompt_progress_fn is not None:
            kwargs["on_prompt_processing_progress"] = prompt_progress_fn

        if stream:
            if "on_prediction_fragment" not in kwargs:
                def default_prediction_fragment(frag):
                    sys.stdout.write(frag.content)
                    sys.stdout.flush()
                kwargs["on_prediction_fragment"] = default_prediction_fragment

        logger.info("Sending chat request to LM Studio...")
        try:
            result = self.model.respond(
                chat,
                **kwargs
            )
            if stream:
                print()
            return result.content
        except Exception as e:
            logger.error(f"LM Studio chat response generation failed: {e}")
            self._model = None
            raise RuntimeError(f"LM Studio chat inference error: {e}") from e
