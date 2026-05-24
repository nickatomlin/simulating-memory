from __future__ import annotations
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .llm import LLM, LLMResponse, LLMToolResponse

# Lone UTF-16 surrogates are invalid in JSON strings; strip/replace before API calls.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
# C0 control chars (except tab/newline/cr) can break gateways or JSON handling; NUL is common culprit.
_CTRL_EXCEPT_WS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_message_text(text: str) -> str:
    if not text:
        return text
    text = _SURROGATE_RE.sub("\ufffd", text)
    text = text.replace("\x00", "")
    text = _CTRL_EXCEPT_WS.sub(" ", text)
    return text


# 1 initial attempt + 5 retries before surfacing the error
_API_RETRIES = 5
_ERROR_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "llm_openai_errors"


def _error_context(exc: Exception) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            ctx["http_status"] = getattr(resp, "status_code", None)
            ctx["response_text"] = resp.text
        except Exception:
            pass
    body = getattr(exc, "body", None)
    if body is not None:
        ctx["body"] = body
    return ctx


def _save_failed_request_log(
    *,
    attempt: int,
    request_kwargs: Dict[str, Any],
    exc: Exception,
) -> Path:
    _ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = _ERROR_LOG_DIR / f"error_{ts}_{uuid.uuid4().hex[:8]}_attempt{attempt}.json"
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "error": _error_context(exc),
        "request_kwargs": request_kwargs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def _api_model_id_for_openai_api(model: str) -> str:
    """Strip ``openai/`` prefix for the official OpenAI API only (not for OpenRouter)."""
    if "/" in model and model.lower().startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _is_openrouter_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    return "openrouter" in str(base_url).lower()


def _token_limit_kwargs(model: str, max_tokens: int) -> Dict[str, Any]:
    """Newer Chat Completions models use max_completion_tokens; max_tokens is deprecated."""
    # Use base model name for detection (works for ``openai/...`` ids too).
    m = _api_model_id_for_openai_api(model).lower()
    if (
        m.startswith("gpt-4o")
        or m.startswith("gpt-5")
        or "gpt-4.1" in m
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    ):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


class OpenAIChatLLM(LLM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        extra_body: Optional[Dict[str, Any]] = None,
        show_request_progress: bool = True,
    ):
        """
        OpenAI-compatible Chat Completions (official API, OpenRouter, Azure, etc.).

        When ``base_url`` points at OpenRouter, ``OPENROUTER_API_KEY`` is preferred; otherwise
        ``OPENAI_API_KEY``. Provider-specific fields (e.g. Qwen ``enable_thinking``) go through
        ``extra_body`` on each request.
        """
        bu = str(base_url).strip() if base_url else None
        if api_key:
            resolved_api_key = api_key
        elif _is_openrouter_base_url(bu):
            resolved_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            resolved_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

        self.client = OpenAI(api_key=resolved_api_key, base_url=bu or None)
        self.model = model
        # OpenRouter expects full route ids (e.g. ``qwen/qwen3-8b``); official OpenAI API strips ``openai/``.
        self._api_model = model if _is_openrouter_base_url(bu) else _api_model_id_for_openai_api(model)
        self._extra_body = dict(extra_body) if extra_body else {}
        self.show_request_progress = show_request_progress
        self.request_count = 0

    def _attach_extra_body(self, request_kwargs: Dict[str, Any]) -> None:
        if not self._extra_body:
            return
        merged = dict(self._extra_body)
        existing = request_kwargs.get("extra_body")
        if isinstance(existing, dict):
            merged = {**existing, **merged}
        request_kwargs["extra_body"] = merged

    def _emit_request_progress(self) -> None:
        if not self.show_request_progress:
            return
        # Single updating line on stderr; \033[K clears to end of line (longer counts).
        sys.stderr.write(f"\rLLM API requests completed: {self.request_count}\033[K")
        sys.stderr.flush()

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = _sanitize_message_text(prompt)
        system = _sanitize_message_text(system) if system else None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self._api_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        request_kwargs.update(_token_limit_kwargs(self.model, max_tokens))
        if seed is not None:
            request_kwargs["seed"] = seed
        self._attach_extra_body(request_kwargs)

        n_attempts = 1 + _API_RETRIES
        last_exc: Optional[Exception] = None
        for attempt_idx in range(n_attempts):
            try:
                resp = self.client.chat.completions.create(**request_kwargs)
                self.request_count += 1
                self._emit_request_progress()
                text = resp.choices[0].message.content or ""
                return LLMResponse(text=text, raw=resp)
            except Exception as e:
                last_exc = e
                log_path = _save_failed_request_log(
                    attempt=attempt_idx + 1,
                    request_kwargs=request_kwargs,
                    exc=e,
                )
                sys.stderr.write(
                    f"\nLLM API error (attempt {attempt_idx + 1}/{n_attempts}); "
                    f"saved request dump to {log_path}\n"
                )
                sys.stderr.flush()
                if attempt_idx < n_attempts - 1:
                    delay = min(2.0**attempt_idx, 30.0)
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 256,
        **kwargs,
    ) -> LLMToolResponse:
        request_kwargs: Dict[str, Any] = {
            "model": self._api_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
        }
        request_kwargs.update(_token_limit_kwargs(self.model, max_tokens))
        self._attach_extra_body(request_kwargs)

        n_attempts = 1 + _API_RETRIES
        last_exc: Optional[Exception] = None
        for attempt_idx in range(n_attempts):
            try:
                resp = self.client.chat.completions.create(**request_kwargs)
                self.request_count += 1
                self._emit_request_progress()
                choice = resp.choices[0]
                msg = choice.message

                tool_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

                return LLMToolResponse(
                    tool_calls=tool_calls,
                    content=msg.content or None,
                    finish_reason=choice.finish_reason or "stop",
                    raw=resp,
                )
            except Exception as e:
                last_exc = e
                log_path = _save_failed_request_log(
                    attempt=attempt_idx + 1,
                    request_kwargs=request_kwargs,
                    exc=e,
                )
                sys.stderr.write(
                    f"\nLLM API error (attempt {attempt_idx + 1}/{n_attempts}); "
                    f"saved request dump to {log_path}\n"
                )
                sys.stderr.flush()
                if attempt_idx < n_attempts - 1:
                    delay = min(2.0**attempt_idx, 30.0)
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc
