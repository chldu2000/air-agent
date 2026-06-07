from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from air_agent.providers import LLMProvider, LLMResponse
from air_agent.skills.skill import Skill

logger = logging.getLogger(__name__)
_delegating_to_legacy_match: ContextVar[bool] = ContextVar(
    "_delegating_to_legacy_match",
    default=False,
)

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


_legacy_match_route_result: ContextVar[SkillRouteResult | None] = ContextVar(
    "_legacy_match_route_result",
    default=None,
)


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


class _ClientBackedRoutingProvider:
    supports_tools = True
    supports_streaming = False

    def __init__(self, client: Any) -> None:
        self._client = client

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=options.get("max_tokens"),
        )
        raw_output = response.choices[0].message.content
        return LLMResponse(content=raw_output or "")


class LLMSkillRouter(SkillRouter):
    """Default implementation: use a lightweight LLM call to select relevant skills."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        model: str = "",
        *,
        client: Any | None = None,
    ) -> None:
        if provider is None:
            if client is None:
                raise TypeError("LLMSkillRouter requires a provider")
            provider = _ClientBackedRoutingProvider(client)

        self._provider = provider
        self._model = model

    async def route(self, user_input: str, skills: list[Skill]) -> SkillRouteResult:
        if (
            type(self).route is LLMSkillRouter.route
            and type(self).match is not LLMSkillRouter.match
            and not _delegating_to_legacy_match.get()
        ):
            delegation_token = _delegating_to_legacy_match.set(True)
            result_token = _legacy_match_route_result.set(None)
            start = perf_counter()
            try:
                matched_skills = await self.match(user_input, skills)
                route_result = _legacy_match_route_result.get()
                if route_result is not None:
                    return route_result
                return SkillRouteResult(
                    matched_skills=matched_skills,
                    duration_ms=_elapsed_ms(start),
                )
            except Exception as exc:
                logger.warning("Skill routing failed", exc_info=True)
                return SkillRouteResult(
                    duration_ms=_elapsed_ms(start),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            finally:
                _legacy_match_route_result.reset(result_token)
                _delegating_to_legacy_match.reset(delegation_token)

        if not skills:
            return SkillRouteResult()

        skill_list = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        messages = [
            {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Available skills:\n{skill_list}\n\nUser input: {user_input}"},
        ]

        start = perf_counter()
        try:
            response = await self._provider.complete(
                model=self._model,
                messages=messages,
                tools=None,
                max_tokens=100,
            )
            raw_output = response.content
            raw_output = raw_output if raw_output is not None else ""
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

            candidate_names = {skill.name for skill in skills}
            matched_skills: list[Skill] = []
            for skill in skills:
                if skill.name in seen_names:
                    matched_skills.append(skill)

            unrecognized_names = [name for name in parsed_names if name not in candidate_names]
            return SkillRouteResult(
                matched_skills=matched_skills,
                raw_output=raw_output,
                duration_ms=_elapsed_ms(start),
                unrecognized_names=unrecognized_names,
            )
        except Exception as exc:
            logger.warning("Skill routing LLM call failed", exc_info=True)
            return SkillRouteResult(
                duration_ms=_elapsed_ms(start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        result = await self.route(user_input, skills)
        if _delegating_to_legacy_match.get():
            _legacy_match_route_result.set(result)
        return result.matched_skills
