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

    async def route(self, user_input: str, skills: list[Skill]) -> SkillRouteResult:
        if not skills:
            return SkillRouteResult()

        skill_list = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        messages = [
            {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Available skills:\n{skill_list}\n\nUser input: {user_input}"},
        ]

        start = perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=100,
            )
            raw_output = response.choices[0].message.content
            raw_output = raw_output if raw_output is not None else ""
        except Exception as exc:
            logger.warning("Skill routing LLM call failed", exc_info=True)
            return SkillRouteResult(
                duration_ms=_elapsed_ms(start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        parsed_output = raw_output.strip().lower()
        if not parsed_output or parsed_output == "none":
            return SkillRouteResult(
                raw_output=raw_output,
                duration_ms=_elapsed_ms(start),
            )

        parsed_names: list[str] = []
        seen_names: set[str] = set()
        for part in parsed_output.split(","):
            name = part.strip().lower()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            parsed_names.append(name)

        candidate_names = {skill.name.lower() for skill in skills}
        matched_skills: list[Skill] = []
        matched_names: set[str] = set()
        for skill in skills:
            normalized_name = skill.name.lower()
            if normalized_name in seen_names and normalized_name not in matched_names:
                matched_skills.append(skill)
                matched_names.add(normalized_name)

        unrecognized_names = [name for name in parsed_names if name not in candidate_names]
        return SkillRouteResult(
            matched_skills=matched_skills,
            raw_output=raw_output,
            duration_ms=_elapsed_ms(start),
            unrecognized_names=unrecognized_names,
        )

    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        return (await self.route(user_input, skills)).matched_skills
