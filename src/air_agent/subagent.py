from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from uuid import uuid4
from typing import Any

from air_agent.config import AgentConfig, SubagentConfig
from air_agent.types import AgentRole, RunEvent, SubagentAggregation, SubagentResult

logger = logging.getLogger(__name__)


async def delegate(
    agent: Any,
    tasks: list[str],
    config: SubagentConfig | None = None,
    *,
    roles: list[AgentRole] | None = None,
    aggregation: SubagentAggregation | Callable | None = None,
) -> list[SubagentResult] | SubagentResult:
    if config is None:
        config = SubagentConfig()

    semaphore = asyncio.Semaphore(config.max_parallel)
    assignments = _assign_roles(tasks, roles)

    async def _run_one(task: str, role: AgentRole | None) -> SubagentResult:
        async with semaphore:
            role_name = role.name if role is not None else None
            events: list[RunEvent] = []
            await agent._emit(
                "subagent_start",
                run_id=f"subagent_{uuid4().hex}",
                conversation_id=None,
                name=role_name,
                content=task,
                metadata={"role": role_name, "task": task},
            )
            try:
                child = _build_child_agent(agent, role, events)
                conversation_id = _subagent_conversation_id(role)
                result = await asyncio.wait_for(
                    child.run(task, conversation_id=conversation_id),
                    timeout=config.timeout,
                )
                subagent_result = SubagentResult(
                    status="success",
                    content=result.content,
                    usage=result.usage,
                    role=role_name,
                    task=task,
                    events=events,
                )
                await agent._emit(
                    "subagent_end",
                    run_id=f"subagent_{uuid4().hex}",
                    conversation_id=None,
                    name=role_name,
                    content=result.content,
                    usage=result.usage,
                    metadata={"role": role_name, "task": task, "status": "success"},
                )
                return subagent_result
            except asyncio.TimeoutError:
                logger.warning("Subagent timed out for task: %s", task[:50])
                result = SubagentResult(
                    status="timeout",
                    content="",
                    role=role_name,
                    task=task,
                    events=events,
                )
                await agent._emit(
                    "subagent_error",
                    run_id=f"subagent_{uuid4().hex}",
                    conversation_id=None,
                    name=role_name,
                    content="timeout",
                    metadata={"role": role_name, "task": task, "status": "timeout"},
                )
                return result
            except Exception as e:
                logger.warning("Subagent error for task: %s — %s", task[:50], e)
                result = SubagentResult(
                    status="error",
                    content=str(e),
                    role=role_name,
                    task=task,
                    events=events,
                    metadata={"error_type": type(e).__name__},
                )
                await agent._emit(
                    "subagent_error",
                    run_id=f"subagent_{uuid4().hex}",
                    conversation_id=None,
                    name=role_name,
                    content=str(e),
                    metadata={
                        "role": role_name,
                        "task": task,
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                return result

    results = list(await asyncio.gather(*[_run_one(task, role) for task, role in assignments]))
    if aggregation is None:
        return results
    return await _aggregate(agent, results, aggregation)


def _assign_roles(
    tasks: list[str],
    roles: list[AgentRole] | None,
) -> list[tuple[str, AgentRole | None]]:
    if not roles:
        return [(task, None) for task in tasks]
    if len(roles) == 1:
        return [(task, roles[0]) for task in tasks]
    if len(tasks) == 1:
        return [(tasks[0], role) for role in roles]
    if len(tasks) == len(roles):
        return list(zip(tasks, roles))
    raise ValueError("roles and tasks must be omitted, one-to-many, many-to-one, or equal length")


def _build_child_agent(parent: Any, role: AgentRole | None, events: list[RunEvent]) -> Any:
    system_prompt = _combine_system_prompts(parent.config.system_prompt, role.system_prompt if role else None)
    handlers = list(parent.config.event_handlers or [])
    handlers.append(events.append)
    child_config = AgentConfig(
        model=parent.config.model,
        api_key=parent.config.api_key,
        base_url=parent.config.base_url,
        system_prompt=system_prompt,
        max_iterations=parent.config.max_iterations,
        tool_timeout=parent.config.tool_timeout,
        mcp_servers=parent.config.mcp_servers,
        default_headers=parent.config.default_headers,
        skills_dir=(role.skills_dir if role and role.skills_dir is not None else parent.config.skills_dir),
        builtin_tools=parent.config.builtin_tools,
        enable_tracing=parent.config.enable_tracing,
        log_events=parent.config.log_events,
        event_handlers=handlers,
        max_tool_retries=parent.config.max_tool_retries,
        provider=parent._provider,
        memory=parent.config.memory,
        memory_enabled=parent.config.memory_enabled,
        memory_search_limit=parent.config.memory_search_limit,
        memory_max_chars=parent.config.memory_max_chars,
        memory_summary_threshold=parent.config.memory_summary_threshold,
        strategy=parent.config.strategy,
        planner=parent.config.planner,
        max_plan_steps=parent.config.max_plan_steps,
    )
    child = parent.__class__(child_config)
    child._registry = parent._registry.clone()
    if role:
        for tool in role.tools:
            child._registry.register(tool)
    return child


def _combine_system_prompts(parent_prompt: str | None, role_prompt: str | None) -> str | None:
    prompts = [prompt for prompt in [parent_prompt, role_prompt] if prompt]
    if not prompts:
        return None
    return "\n\n".join(prompts)


def _subagent_conversation_id(role: AgentRole | None) -> str:
    if role and role.memory_scope:
        return role.memory_scope
    role_name = role.name if role else "default"
    return f"subagent:{role_name}:{uuid4().hex}"


async def _aggregate(
    agent: Any,
    results: list[SubagentResult],
    aggregation: SubagentAggregation | Callable,
) -> SubagentResult:
    if aggregation == "concat":
        status = "success" if all(result.status == "success" for result in results) else "error"
        content = "\n\n".join(
            _format_result(result, index)
            for index, result in enumerate(results)
        )
        return SubagentResult(
            status=status,
            content=content,
            metadata={"aggregation": "concat", "count": len(results)},
        )
    if aggregation == "summarize":
        return await _summarize_results(agent, results)
    if aggregation == "vote":
        return await _vote_results(agent, results)
    if callable(aggregation):
        value = aggregation(results)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, SubagentResult):
            return value
        return SubagentResult(status="success", content=str(value), metadata={"aggregation": "callable"})
    return SubagentResult(status="error", content=f"Unsupported aggregation: {aggregation}", metadata={"aggregation": str(aggregation)})


def _format_result(result: SubagentResult, index: int) -> str:
    role = result.role or "default"
    task = result.task or ""
    return f"[{index}] role={role} status={result.status} task={task}\n{result.content}"


async def _summarize_results(agent: Any, results: list[SubagentResult]) -> SubagentResult:
    response = await agent._provider.complete(
        model=agent.config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following subagent results for the user. "
                    "Explicitly mention failed or timed out subagents."
                ),
            },
            {"role": "user", "content": _aggregation_prompt(results)},
        ],
        tools=None,
    )
    status = "success" if all(result.status == "success" for result in results) else "error"
    return SubagentResult(
        status=status,
        content=response.content or "",
        usage=response.usage,
        metadata={"aggregation": "summarize", "count": len(results)},
    )


async def _vote_results(agent: Any, results: list[SubagentResult]) -> SubagentResult:
    try:
        response = await agent._provider.complete(
            model=agent.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Choose the best subagent result. Return strict JSON only: "
                        '{"winner_index": 0, "reason": "..."}'
                    ),
                },
                {"role": "user", "content": _aggregation_prompt(results)},
            ],
            tools=None,
        )
        data = json.loads((response.content or "").strip())
        winner_index = data.get("winner_index")
        reason = data.get("reason", "")
        if not isinstance(winner_index, int) or not 0 <= winner_index < len(results):
            raise ValueError(f"winner_index out of range: {winner_index}")
        winner = results[winner_index]
        return SubagentResult(
            status=winner.status,
            content=winner.content,
            usage=winner.usage,
            role=winner.role,
            task=winner.task,
            events=list(winner.events),
            metadata={
                **winner.metadata,
                "aggregation": "vote",
                "winner_index": winner_index,
                "reason": reason,
            },
        )
    except Exception as exc:
        return SubagentResult(
            status="error",
            content=f"vote aggregation failed: {str(exc) or type(exc).__name__}",
            metadata={"aggregation": "vote", "error_type": type(exc).__name__},
        )


def _aggregation_prompt(results: list[SubagentResult]) -> str:
    lines = ["Subagent results:"]
    for index, result in enumerate(results):
        lines.append(_format_result(result, index))
    return "\n\n".join(lines)
