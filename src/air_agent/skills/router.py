from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from air_agent.skills.skill import Skill

logger = logging.getLogger(__name__)

_ROUTING_SYSTEM_PROMPT = """\
You are a skill router. Given a user input and a list of available skills, \
return ONLY the names of skills that are relevant, separated by commas. \
If no skills are relevant, respond with "none". \
Do not include any other text or explanation."""


@dataclass
class SkillRouteResult:
    matched_skills: list[Skill] = field(default_factory=list)
    raw_output: str = ""
    duration_ms: float | None = None
    unrecognized_names: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 3)


class SkillRouter(ABC):
    @abstractmethod
    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        """Match user input against available skills."""

    async def route(self, user_input: str, skills: list[Skill]) -> SkillRouteResult:
        start = perf_counter()
        try:
            matched_skills = await self.match(user_input, skills)
        except Exception as exc:
            logger.warning("Skill routing failed", exc_info=True)
            return SkillRouteResult(
                duration_ms=_elapsed_ms(start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return SkillRouteResult(
            matched_skills=matched_skills,
            duration_ms=_elapsed_ms(start),
        )


class LLMSkillRouter(SkillRouter):
    """Default implementation: use a lightweight LLM call to select relevant skills."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        if not skills:
            return []

        skill_list = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        messages = [
            {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Available skills:\n{skill_list}\n\nUser input: {user_input}"},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip().lower()
        except Exception:
            logger.warning("Skill routing LLM call failed", exc_info=True)
            return []

        if content == "none" or not content:
            return []

        matched_names = {name.strip() for name in content.split(",")}
        return [s for s in skills if s.name in matched_names]
