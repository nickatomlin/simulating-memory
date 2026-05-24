from __future__ import annotations
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from .llm import LLM, LLMResponse, LLMToolResponse

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_CTRL_EXCEPT_WS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_message_text(text: str) -> str:
    if not text:
        return text
    text = _SURROGATE_RE.sub("�", text)
    text = text.replace("\x00", "")
    text = _CTRL_EXCEPT_WS.sub(" ", text)
    return text


_API_RETRIES = 5
_ERROR_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "llm_anthropic_errors"


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


def _save_failed_request_log(*, attempt: int, request_kwargs: Dict[str, Any], exc: Exception) -> Path:
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


def _api_model_id(model: str) -> str:
    """Strip provider prefix like ``anthropic/claude-opus-4-6`` -> ``claude-opus-4-6``."""
    if "/" in model and model.lower().startswith("anthropic/"):
        return model.split("/", 1)[1]
    return model


def _convert_tools_openai_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI tool schema -> Anthropic tool schema."""
    out: List[Dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _convert_tool_choice(tool_choice: str) -> Dict[str, Any]:
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice == "any" or tool_choice == "required":
        return {"type": "any"}
    return {"type": "auto"}


def _convert_messages_openai_to_anthropic(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Translate OpenAI-style messages into (system_text, anthropic_messages).

    OpenAI shape used by wm_agent:
      - {"role": "system", "content": str}
      - {"role": "user", "content": str}
      - {"role": "assistant", "content": str|"", "tool_calls": [{"id","type","function":{"name","arguments"}}]}
      - {"role": "tool", "tool_call_id": str, "content": str}
    """
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []

    def _flush_pending_tool_results(buf: List[Dict[str, Any]]):
        if buf:
            out.append({"role": "user", "content": buf[:]})
            buf.clear()

    pending_tool_results: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            content = m.get("content") or ""
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", "") or "",
            })
            continue

        # Any non-tool message: flush any buffered tool_results as a user turn first.
        _flush_pending_tool_results(pending_tool_results)

        if role == "user":
            content = m.get("content")
            if isinstance(content, list):
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user", "content": [{"type": "text", "text": content or "(empty)"}]})
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            content = m.get("content")
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend([b for b in content if not (isinstance(b, dict) and b.get("type") == "text" and not b.get("text"))])
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                    "name": fn.get("name", ""),
                    "input": args or {},
                })
            if not blocks:
                blocks = [{"type": "text", "text": "(no response)"}]
            out.append({"role": "assistant", "content": blocks})

    _flush_pending_tool_results(pending_tool_results)

    # Anthropic requires alternating user/assistant roles. Merge consecutive same-role messages.
    merged: List[Dict[str, Any]] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = list(merged[-1]["content"]) + list(msg["content"])
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, merged


def _extract_text_and_tool_calls(resp: Any) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in resp.content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input or {}),
                },
            })
    return "".join(text_parts), tool_calls


class AnthropicChatLLM(LLM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        show_request_progress: bool = True,
    ):
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        client_kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self._api_model = _api_model_id(model)
        self.show_request_progress = show_request_progress
        self.request_count = 0

    def _emit_request_progress(self) -> None:
        if not self.show_request_progress:
            return
        sys.stderr.write(f"\rLLM API requests completed: {self.request_count}\033[K")
        sys.stderr.flush()

    def _create_with_retry(self, request_kwargs: Dict[str, Any]) -> Any:
        n_attempts = 1 + _API_RETRIES
        last_exc: Optional[Exception] = None
        for attempt_idx in range(n_attempts):
            try:
                resp = self.client.messages.create(**request_kwargs)
                self.request_count += 1
                self._emit_request_progress()
                return resp
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

        request_kwargs: Dict[str, Any] = {
            "model": self._api_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Anthropic rejects sending both temperature and top_p; only send top_p when non-default.
        if top_p is not None and top_p != 1.0:
            request_kwargs["top_p"] = top_p
        if system:
            request_kwargs["system"] = system

        resp = self._create_with_retry(request_kwargs)
        text, _ = _extract_text_and_tool_calls(resp)
        return LLMResponse(text=text, raw=resp)

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
        system_text, anth_messages = _convert_messages_openai_to_anthropic(messages)
        anth_tools = _convert_tools_openai_to_anthropic(tools)
        anth_tool_choice = _convert_tool_choice(tool_choice)

        request_kwargs: Dict[str, Any] = {
            "model": self._api_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anth_messages,
            "tools": anth_tools,
            "tool_choice": anth_tool_choice,
        }
        if system_text:
            request_kwargs["system"] = system_text

        resp = self._create_with_retry(request_kwargs)
        text, tool_calls = _extract_text_and_tool_calls(resp)
        finish_reason = getattr(resp, "stop_reason", "stop") or "stop"
        return LLMToolResponse(
            tool_calls=tool_calls,
            content=text or None,
            finish_reason=finish_reason,
            raw=resp,
        )
