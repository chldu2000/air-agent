from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from air_agent.types import TokenUsage


@dataclass(slots=True)
class LLMToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None


@dataclass(slots=True)
class LLMStreamToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(slots=True)
class LLMStreamChunk:
    content_delta: str = ""
    tool_call_deltas: list[LLMStreamToolCallDelta] = field(default_factory=list)
    usage: TokenUsage | None = None


@runtime_checkable
class LLMProvider(Protocol):
    supports_tools: bool
    supports_streaming: bool

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]: ...


__all__ = [
    "LLMToolCall",
    "LLMResponse",
    "LLMStreamToolCallDelta",
    "LLMStreamChunk",
    "LLMProvider",
    "TokenUsage",
]
