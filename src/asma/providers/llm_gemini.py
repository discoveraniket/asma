import os
import logging
from types import SimpleNamespace
from typing import Callable, Optional
from asma.interfaces.llm import LLMProvider

logger = logging.getLogger(__name__)

AVAILABLE_GEMINI_MODELS = [
    {"id": "gemini-3.5-flash-lite", "label": "gemini-3.5-flash-lite (Recommended)", "is_default": True},
    {"id": "gemini-2.5-flash", "label": "gemini-2.5-flash (Ultra Fast)"},
    {"id": "gemini-2.0-flash", "label": "gemini-2.0-flash (Fast)"},
    {"id": "gemini-2.0-flash-lite", "label": "gemini-2.0-flash-lite (Cost-Effective Lite)"},
    {"id": "gemini-1.5-flash", "label": "gemini-1.5-flash (Standard)"},
    {"id": "gemini-1.5-pro", "label": "gemini-1.5-pro (High Capacity)"},
]

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

class GeminiProvider(LLMProvider):
    """
    Client wrapper for Google Gemini Cloud API to perform inference.
    Supports official google-genai SDK.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        enable_thinking: bool = True
    ):
        self.api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self.enable_thinking = enable_thinking
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Google Gemini API Key is missing. "
                    "Please set the GEMINI_API_KEY environment variable or enter your API key in Settings."
                )
            try:
                from google import genai
                logger.info(f"Connecting to Google Gemini API (model: {self.model_name}, thinking: {self.enable_thinking})...")
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini client: {e}")
                raise RuntimeError(f"Could not connect to Google Gemini API: {e}") from e
        return self._client

    def _get_thinking_config(self, types_module):
        if self.enable_thinking:
            try:
                return types_module.ThinkingConfig(include_thoughts=True)
            except Exception:
                return None
        else:
            try:
                return types_module.ThinkingConfig(thinking_budget=0)
            except Exception:
                return None

    def respond(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        from google.genai import types

        on_frag = kwargs.pop("on_prediction_fragment", None)
        kwargs.pop("on_prompt_processing_progress", None)
        kwargs.pop("ignore_context_limit", None)

        thinking_config = self._get_thinking_config(types)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            thinking_config=thinking_config
        )

        full_response = ""
        state = {"in_thought": False}
        try:
            if stream:
                response = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                for chunk in response:
                    full_response += self._process_chunk(chunk, on_frag, state)
                if state["in_thought"]:
                    closing = "\n</think>\n"
                    full_response += closing
                    if on_frag:
                        on_frag(SimpleNamespace(content=closing))
                    state["in_thought"] = False
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                full_response = self._process_non_stream_response(response)
        except Exception as err:
            logger.error(f"Gemini API generation error: {err}")
            raise RuntimeError(f"Gemini API inference failed: {err}") from err

        return full_response

    def respond_chat(
        self,
        chat_history: list,
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        from google.genai import types

        on_frag = kwargs.pop("on_prediction_fragment", None)
        kwargs.pop("on_prompt_processing_progress", None)
        kwargs.pop("ignore_context_limit", None)

        system_instruction = None
        contents = []

        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)]
                    )
                )
            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)]
                    )
                )

        thinking_config = self._get_thinking_config(types)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            thinking_config=thinking_config
        )

        full_response = ""
        state = {"in_thought": False}
        try:
            if stream:
                response = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                for chunk in response:
                    full_response += self._process_chunk(chunk, on_frag, state)
                if state["in_thought"]:
                    closing = "\n</think>\n"
                    full_response += closing
                    if on_frag:
                        on_frag(SimpleNamespace(content=closing))
                    state["in_thought"] = False
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                full_response = self._process_non_stream_response(response)
        except Exception as err:
            logger.error(f"Gemini API chat generation error: {err}")
            raise RuntimeError(f"Gemini API chat inference failed: {err}") from err

        return full_response

    def _process_chunk(self, chunk, on_frag, state=None):
        if state is None:
            state = {"in_thought": False}

        chunk_str = ""
        has_parts = False

        if hasattr(chunk, "candidates") and chunk.candidates:
            cand = chunk.candidates[0]
            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts") and cand.content.parts:
                has_parts = True
                for part in cand.content.parts:
                    part_text = getattr(part, "text", "") or ""
                    if not part_text:
                        continue
                    is_thought = getattr(part, "thought", False)
                    if is_thought:
                        if not state["in_thought"]:
                            open_tag = "<think>\n"
                            chunk_str += open_tag
                            if on_frag:
                                on_frag(SimpleNamespace(content=open_tag))
                            state["in_thought"] = True
                        chunk_str += part_text
                        if on_frag:
                            on_frag(SimpleNamespace(content=part_text))
                    else:
                        if state["in_thought"]:
                            close_tag = "\n</think>\n"
                            chunk_str += close_tag
                            if on_frag:
                                on_frag(SimpleNamespace(content=close_tag))
                            state["in_thought"] = False
                        chunk_str += part_text
                        if on_frag:
                            on_frag(SimpleNamespace(content=part_text))

        if not has_parts:
            text_chunk = getattr(chunk, "text", "") or ""
            if text_chunk:
                if state["in_thought"]:
                    close_tag = "\n</think>\n"
                    chunk_str += close_tag
                    if on_frag:
                        on_frag(SimpleNamespace(content=close_tag))
                    state["in_thought"] = False
                chunk_str += text_chunk
                if on_frag:
                    on_frag(SimpleNamespace(content=text_chunk))

        return chunk_str

    def _process_non_stream_response(self, response):
        thought_parts = []
        answer_parts = []
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts") and cand.content.parts:
                for part in cand.content.parts:
                    part_text = getattr(part, "text", "") or ""
                    if not part_text:
                        continue
                    is_thought = getattr(part, "thought", False)
                    if is_thought:
                        thought_parts.append(part_text)
                    else:
                        answer_parts.append(part_text)

        res = ""
        if thought_parts:
            res += f"<think>\n{''.join(thought_parts)}\n</think>\n"
        res += ''.join(answer_parts)
        if not res:
            res = getattr(response, "text", "") or ""
        return res
