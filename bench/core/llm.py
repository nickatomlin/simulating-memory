from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class LLMResponse:
    text: str
    raw: Any = None

@dataclass
class LLMToolResponse:
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    content: Optional[str] = None
    finish_reason: str = "stop"
    raw: Any = None

class LLM:
    def generate(self, prompt: str, *, system: Optional[str]=None, **kwargs) -> LLMResponse:
        raise NotImplementedError

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
        raise NotImplementedError
