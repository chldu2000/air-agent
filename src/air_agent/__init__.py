"""Air Agent — lightweight AI agent library."""

from air_agent.config import AgentConfig, MCPServerStdio, MCPServerSSE, SubagentConfig
from air_agent.providers import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamToolCallDelta,
    LLMToolCall,
)
from air_agent.types import (
    Response,
    RunEvent,
    StreamEvent,
    SubagentResult,
    ToolErrorKind,
    ToolExecutionResult,
)
from air_agent.agent import Agent
from air_agent.skills.skill import Skill
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import SkillRouteResult, SkillRouter, LLMSkillRouter
from air_agent.tools.builtin.config import BuiltinToolsConfig

__all__ = [
    "Agent",
    "AgentConfig",
    "MCPServerStdio",
    "MCPServerSSE",
    "SubagentConfig",
    "Response",
    "RunEvent",
    "StreamEvent",
    "SubagentResult",
    "ToolErrorKind",
    "ToolExecutionResult",
    "Skill",
    "SkillManager",
    "SkillRouteResult",
    "SkillRouter",
    "LLMSkillRouter",
    "BuiltinToolsConfig",
    "LLMToolCall",
    "LLMResponse",
    "LLMStreamToolCallDelta",
    "LLMStreamChunk",
    "LLMProvider",
]
