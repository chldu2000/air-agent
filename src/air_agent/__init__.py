"""Air Agent — lightweight AI agent library."""

from air_agent.config import AgentConfig, MCPServerStdio, MCPServerSSE, SubagentConfig
from air_agent.types import Response, StreamEvent, SubagentResult
from air_agent.agent import Agent
from air_agent.skills.skill import Skill
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import SkillRouter, LLMSkillRouter

__all__ = [
    "Agent",
    "AgentConfig",
    "MCPServerStdio",
    "MCPServerSSE",
    "SubagentConfig",
    "Response",
    "StreamEvent",
    "SubagentResult",
    "Skill",
    "SkillManager",
    "SkillRouter",
    "LLMSkillRouter",
]