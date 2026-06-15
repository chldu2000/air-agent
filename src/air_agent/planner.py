from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from air_agent.providers import LLMProvider

PlanStepStatus = Literal["pending", "running", "success", "error", "skipped"]
PlanStatus = Literal["pending", "running", "success", "error", "revised"]
StepResultStatus = Literal["success", "error", "skipped"]


@dataclass(slots=True)
class PlanStep:
    id: str
    description: str
    status: PlanStepStatus = "pending"
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class StepResult:
    step_id: str
    status: StepResultStatus
    content: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "content": self.content,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Plan:
    goal: str
    steps: list[PlanStep]
    status: PlanStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        ids: set[str] = set()
        for step in self.steps:
            if not step.id:
                raise ValueError("plan step id is required")
            if step.id in ids:
                raise ValueError(f"duplicate step id: {step.id}")
            ids.add(step.id)

        for step in self.steps:
            for dependency in step.dependencies:
                if dependency not in ids:
                    raise ValueError(f"step {step.id} has unknown dependency: {dependency}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class PlanContext:
    goal: str
    messages: list[dict[str, Any]]
    conversation_id: str | None = None
    previous_results: list[StepResult] = field(default_factory=list)
    run_step: Callable[[PlanStep], Awaitable[StepResult]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Planner(Protocol):
    async def create_plan(self, goal: str, context: PlanContext) -> Plan: ...

    async def execute_step(self, step: PlanStep, context: PlanContext) -> StepResult: ...

    async def revise_plan(self, plan: Plan, result: StepResult) -> Plan: ...


class LLMPlanner:
    def __init__(self, *, provider: LLMProvider, model: str, max_steps: int = 8):
        if max_steps < 1:
            raise ValueError("max_steps must be greater than 0")
        self.provider = provider
        self.model = model
        self.max_steps = max_steps

    async def create_plan(self, goal: str, context: PlanContext) -> Plan:
        response = await self.provider.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a planning component. Return only strict JSON with this shape: "
                        '{"steps":[{"id":"step_1","description":"...","dependencies":[]}]}. '
                        "Use at most the requested number of steps. Do not include markdown unless fenced JSON is unavoidable."
                    ),
                },
                {
                    "role": "user",
                    "content": self._plan_prompt(goal, context),
                },
            ],
            tools=None,
        )
        data = self._parse_plan_json(response.content)
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("planner response must contain a steps list")
        if not raw_steps:
            raise ValueError("planner response must contain at least one step")

        steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps[: self.max_steps], start=1):
            if not isinstance(raw_step, dict):
                raise ValueError(f"plan step {index} must be an object")
            step_id = raw_step.get("id")
            description = raw_step.get("description")
            dependencies = raw_step.get("dependencies", [])
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError(f"plan step {index} must include a non-empty id")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"plan step {step_id} must include a non-empty description")
            if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
                raise ValueError(f"plan step {step_id} dependencies must be a list of strings")
            steps.append(
                PlanStep(
                    id=step_id.strip(),
                    description=description.strip(),
                    dependencies=list(dependencies),
                    metadata={"source_index": index - 1},
                )
            )

        plan = Plan(goal=goal, steps=steps, metadata={"planner": self.__class__.__name__})
        plan.validate()
        return plan

    async def execute_step(self, step: PlanStep, context: PlanContext) -> StepResult:
        if context.run_step is None:
            raise RuntimeError("PlanContext.run_step is required to execute a plan step")
        return await context.run_step(step)

    async def revise_plan(self, plan: Plan, result: StepResult) -> Plan:
        if result.status != "error":
            return plan

        failed = result.step_id
        for step in plan.steps:
            if step.id == failed:
                step.status = "error"

        blocked = {failed}
        changed = True
        while changed:
            changed = False
            for step in plan.steps:
                if step.id in blocked:
                    continue
                if any(dependency in blocked for dependency in step.dependencies):
                    step.status = "skipped"
                    blocked.add(step.id)
                    changed = True

        plan.status = "revised"
        return plan

    def _plan_prompt(self, goal: str, context: PlanContext) -> str:
        conversation = json.dumps(context.messages[-8:], ensure_ascii=False)
        return (
            f"Goal:\n{goal}\n\n"
            f"Maximum steps: {self.max_steps}\n\n"
            "Recent conversation/messages for context:\n"
            f"{conversation}\n\n"
            "Return a JSON object only."
        )

    def _parse_plan_json(self, content: str) -> dict[str, Any]:
        raw = _extract_json_object(content)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"planner response is not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("planner response must be a JSON object")
        return data


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return stripped


__all__ = [
    "LLMPlanner",
    "Plan",
    "PlanContext",
    "Planner",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "StepResult",
    "StepResultStatus",
]
