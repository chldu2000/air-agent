from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from air_agent.tools.builtin.config import BuiltinToolsConfig


@dataclass
class MCPServerStdio:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass
class MCPServerSSE:
    url: str
    headers: dict[str, str] | None = None


@dataclass
class SubagentConfig:
    max_parallel: int = 5
    timeout: float = 60.0
    inherit_tools: bool = True


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_mcp_server(data: dict[str, Any]) -> MCPServerStdio | MCPServerSSE:
    if "command" in data:
        return MCPServerStdio(
            command=data["command"],
            args=data.get("args", []),
            env=data.get("env"),
        )
    if "url" in data:
        return MCPServerSSE(
            url=data["url"],
            headers=data.get("headers"),
        )
    raise ValueError(
        f"Cannot determine MCP server type: need 'command' (stdio) or 'url' (sse), got keys {list(data.keys())}"
    )


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    max_iterations: int = 20
    tool_timeout: float = 30.0
    mcp_servers: list[MCPServerStdio | MCPServerSSE] = field(default_factory=list)
    default_headers: dict[str, str] | None = None
    skills_dir: str | None = None
    builtin_tools: BuiltinToolsConfig | None = None
    enable_tracing: bool = False
    log_events: bool = False
    event_handlers: list[Callable[[Any], Any]] | None = None
    max_tool_retries: int = 0
    provider: Any = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    @classmethod
    def from_json(cls, path: str) -> AgentConfig:
        with open(path) as f:
            data = json.load(f)

        mcp_servers = [_parse_mcp_server(s) for s in data.pop("mcp_servers", [])]
        builtin_raw = data.pop("builtin_tools", None)
        provider_raw = data.get("provider")
        if "provider" in data and provider_raw is not None and not isinstance(provider_raw, str):
            raise ValueError("provider must be a string or null")
        field_names = {
            f.name for f in cls.__dataclass_fields__.values() if f.name != "event_handlers"
        }
        kwargs = {k: v for k, v in data.items() if k in field_names}

        if builtin_raw and isinstance(builtin_raw, dict):
            kwargs["builtin_tools"] = BuiltinToolsConfig.from_dict(builtin_raw)

        return cls(mcp_servers=mcp_servers, **kwargs)

    @classmethod
    def from_env(cls, prefix: str = "AIR_") -> AgentConfig:
        kwargs: dict[str, Any] = {}

        env_map: dict[str, tuple[str, type]] = {
            f"{prefix}MODEL": ("model", str),
            f"{prefix}API_KEY": ("api_key", str),
            f"{prefix}BASE_URL": ("base_url", str),
            f"{prefix}PROVIDER": ("provider", str),
            f"{prefix}SYSTEM_PROMPT": ("system_prompt", str),
            f"{prefix}MAX_ITERATIONS": ("max_iterations", int),
            f"{prefix}TOOL_TIMEOUT": ("tool_timeout", float),
            f"{prefix}SKILLS_DIR": ("skills_dir", str),
            f"{prefix}ENABLE_TRACING": ("enable_tracing", _parse_bool),
            f"{prefix}LOG_EVENTS": ("log_events", _parse_bool),
            f"{prefix}MAX_TOOL_RETRIES": ("max_tool_retries", int),
        }

        for env_key, (field_name, type_) in env_map.items():
            value = os.environ.get(env_key)
            if value is not None:
                kwargs[field_name] = type_(value)

        mcp_raw = os.environ.get(f"{prefix}MCP_SERVERS")
        if mcp_raw:
            kwargs["mcp_servers"] = [_parse_mcp_server(s) for s in json.loads(mcp_raw)]

        headers_raw = os.environ.get(f"{prefix}DEFAULT_HEADERS")
        if headers_raw:
            kwargs["default_headers"] = json.loads(headers_raw)

        builtin_raw = os.environ.get(f"{prefix}BUILTIN_TOOLS")
        if builtin_raw:
            kwargs["builtin_tools"] = BuiltinToolsConfig.from_dict(json.loads(builtin_raw))

        return cls(**kwargs)
