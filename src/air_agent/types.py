from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal


ToolErrorKind = Literal[
    "invalid_arguments",
    "tool_not_found",
    "timeout",
    "permission_denied",
    "tool_error",
]
SubagentAggregation = Literal["concat", "summarize", "vote"]


@dataclass
class AgentRole:
    name: str
    description: str = ""
    system_prompt: str | None = None
    tools: list[Callable] = field(default_factory=list)
    skills_dir: str | None = None
    memory_scope: str | None = None


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Response:
    content: str
    usage: TokenUsage | dict[str, int] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)


@dataclass
class StreamEvent:
    type: str  # "text" | "tool_call" | "tool_result" | "done"
    content: str = ""
    name: str | None = None
    arguments: str | None = None
    usage: TokenUsage | dict[str, int] | None = None

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)


@dataclass
class RunEvent:
    type: str
    run_id: str
    conversation_id: str | None = None
    iteration: int | None = None
    timestamp: datetime | None = None
    name: str | None = None
    arguments: str | None = None
    content: str | None = None
    duration_ms: float | None = None
    usage: TokenUsage | dict[str, int] | None = None
    error_kind: ToolErrorKind | None = None
    attempt: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "run_id": self.run_id,
        }
        optional_fields = {
            "conversation_id": self.conversation_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "name": self.name,
            "arguments": self.arguments,
            "duration_ms": self.duration_ms,
            "error_kind": self.error_kind,
            "attempt": self.attempt,
        }
        data.update(
            {key: value for key, value in optional_fields.items() if value is not None}
        )
        if self.content is not None:
            data["content"] = self.content
        if self.usage is not None:
            data["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class ToolExecutionResult:
    ok: bool
    content: str
    error_kind: ToolErrorKind | None = None
    duration_ms: float | None = None

    @classmethod
    def success(cls, content: str, duration_ms: float | None = None) -> ToolExecutionResult:
        return cls(ok=True, content=content, duration_ms=duration_ms)

    @classmethod
    def failure(
        cls,
        content: str,
        error_kind: ToolErrorKind,
        duration_ms: float | None = None,
    ) -> ToolExecutionResult:
        return cls(ok=False, content=content, error_kind=error_kind, duration_ms=duration_ms)


@dataclass
class SubagentResult:
    status: str  # "success" | "timeout" | "error"
    content: str
    usage: TokenUsage | dict[str, int] | None = None
    role: str | None = None
    task: str | None = None
    events: list[RunEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)
