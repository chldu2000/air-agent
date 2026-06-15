from __future__ import annotations

from typing import Any

import pytest

from air_agent import LLMPlanner, Plan, PlanContext, Planner, PlanStep, StepResult
from air_agent.providers import LLMResponse


class FakeProvider:
    supports_tools = True
    supports_streaming = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakePlanner:
    async def create_plan(self, goal: str, context: PlanContext) -> Plan:
        return Plan(goal=goal, steps=[PlanStep(id="step_1", description="Do it")])

    async def execute_step(self, step: PlanStep, context: PlanContext) -> StepResult:
        return StepResult(step_id=step.id, status="success", content="done")

    async def revise_plan(self, plan: Plan, result: StepResult) -> Plan:
        return plan


def test_plan_step_defaults_are_serialization_friendly():
    step = PlanStep(id="step_1", description="Gather facts")

    assert step.id == "step_1"
    assert step.description == "Gather facts"
    assert step.status == "pending"
    assert step.dependencies == []
    assert step.metadata == {}
    assert step.to_dict() == {
        "id": "step_1",
        "description": "Gather facts",
        "status": "pending",
        "dependencies": [],
        "metadata": {},
    }


def test_plan_defaults_and_dependency_validation():
    plan = Plan(
        goal="Ship v0.6",
        steps=[
            PlanStep(id="step_1", description="First"),
            PlanStep(id="step_2", description="Second", dependencies=["step_1"]),
        ],
    )

    assert plan.status == "pending"
    assert plan.created_at.tzinfo is not None
    plan.validate()


def test_plan_validation_rejects_unknown_dependencies():
    plan = Plan(
        goal="Bad plan",
        steps=[PlanStep(id="step_1", description="Only", dependencies=["missing"])],
    )

    with pytest.raises(ValueError, match="unknown dependency"):
        plan.validate()


@pytest.mark.asyncio
async def test_fake_planner_satisfies_protocol():
    planner: Planner = FakePlanner()
    context = PlanContext(goal="Goal", messages=[], run_step=lambda step: None)

    plan = await planner.create_plan("Goal", context)
    result = await planner.execute_step(plan.steps[0], context)
    revised = await planner.revise_plan(plan, result)

    assert plan.steps[0].id == "step_1"
    assert result.status == "success"
    assert revised is plan


@pytest.mark.asyncio
async def test_llm_planner_creates_plan_from_json():
    provider = FakeProvider(
        [
            LLMResponse(
                content='{"steps":[{"id":"step_1","description":"Gather facts","dependencies":[]},{"id":"step_2","description":"Summarize","dependencies":["step_1"]}]}'
            )
        ]
    )
    planner = LLMPlanner(provider=provider, model="fake", max_steps=8)

    plan = await planner.create_plan("Write a report", PlanContext(goal="Write a report", messages=[]))

    assert [step.id for step in plan.steps] == ["step_1", "step_2"]
    assert plan.steps[1].dependencies == ["step_1"]
    assert provider.calls[0]["tools"] is None


@pytest.mark.asyncio
async def test_llm_planner_accepts_fenced_json():
    provider = FakeProvider([
        LLMResponse(content='```json\n{"steps":[{"id":"step_1","description":"Do it","dependencies":[]}]}\n```')
    ])
    planner = LLMPlanner(provider=provider, model="fake")

    plan = await planner.create_plan("Goal", PlanContext(goal="Goal", messages=[]))

    assert plan.steps[0].id == "step_1"


@pytest.mark.asyncio
async def test_llm_planner_rejects_malformed_json():
    provider = FakeProvider([LLMResponse(content="not json")])
    planner = LLMPlanner(provider=provider, model="fake")

    with pytest.raises(ValueError, match="not valid JSON"):
        await planner.create_plan("Goal", PlanContext(goal="Goal", messages=[]))


@pytest.mark.asyncio
async def test_llm_planner_rejects_empty_steps():
    provider = FakeProvider([LLMResponse(content='{"steps": []}')])
    planner = LLMPlanner(provider=provider, model="fake")

    with pytest.raises(ValueError, match="at least one step"):
        await planner.create_plan("Goal", PlanContext(goal="Goal", messages=[]))


@pytest.mark.asyncio
async def test_llm_planner_validates_dependencies():
    provider = FakeProvider([
        LLMResponse(content='{"steps":[{"id":"step_1","description":"Only","dependencies":["missing"]}]}')
    ])
    planner = LLMPlanner(provider=provider, model="fake")

    with pytest.raises(ValueError, match="unknown dependency"):
        await planner.create_plan("Goal", PlanContext(goal="Goal", messages=[]))


@pytest.mark.asyncio
async def test_llm_planner_truncates_to_max_steps():
    provider = FakeProvider([
        LLMResponse(
            content='{"steps":[{"id":"step_1","description":"One"},{"id":"step_2","description":"Two"},{"id":"step_3","description":"Three"}]}'
        )
    ])
    planner = LLMPlanner(provider=provider, model="fake", max_steps=2)

    plan = await planner.create_plan("Goal", PlanContext(goal="Goal", messages=[]))

    assert [step.id for step in plan.steps] == ["step_1", "step_2"]


@pytest.mark.asyncio
async def test_llm_planner_execute_step_uses_context_callback():
    provider = FakeProvider([])
    planner = LLMPlanner(provider=provider, model="fake")
    step = PlanStep(id="step_1", description="Do it")

    async def run_step(callback_step: PlanStep) -> StepResult:
        assert callback_step is step
        return StepResult(step_id=callback_step.id, status="success", content="ok")

    result = await planner.execute_step(step, PlanContext(goal="Goal", messages=[], run_step=run_step))

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_llm_planner_execute_step_requires_callback():
    provider = FakeProvider([])
    planner = LLMPlanner(provider=provider, model="fake")

    with pytest.raises(RuntimeError, match="run_step is required"):
        await planner.execute_step(PlanStep(id="step_1", description="Do it"), PlanContext(goal="Goal", messages=[]))


@pytest.mark.asyncio
async def test_llm_planner_revise_plan_marks_failed_and_blocked_steps():
    provider = FakeProvider([])
    planner = LLMPlanner(provider=provider, model="fake")
    plan = Plan(
        goal="Goal",
        steps=[
            PlanStep(id="step_1", description="First"),
            PlanStep(id="step_2", description="Second", dependencies=["step_1"]),
            PlanStep(id="step_3", description="Third", dependencies=["step_2"]),
        ],
    )

    revised = await planner.revise_plan(plan, StepResult(step_id="step_1", status="error", error="boom"))

    assert revised.status == "revised"
    assert [step.status for step in revised.steps] == ["error", "skipped", "skipped"]
