import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from air_agent import AgentRole
from air_agent.agent import Agent
from air_agent.config import AgentConfig, SubagentConfig
from air_agent.memory import InMemoryMemoryStore, MemoryRecord
from air_agent.providers import LLMResponse, LLMToolCall
from air_agent.types import SubagentResult


def _mock_response(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
    return resp


class FakeCompletionProvider:
    supports_tools = True
    supports_streaming = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
                "tools": list(kwargs["tools"]) if kwargs.get("tools") else None,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_delegate_parallel():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _mock_response("done")
        results = await agent.delegate(
            tasks=["Task A", "Task B"],
            config=SubagentConfig(max_parallel=2, timeout=10),
        )

    assert len(results) == 2
    assert all(isinstance(r, SubagentResult) for r in results)
    assert all(r.status == "success" for r in results)


@pytest.mark.asyncio
async def test_delegate_timeout():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    async def _slow_response(*args, **kwargs):
        import asyncio
        await asyncio.sleep(10)
        return _mock_response("done")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = _slow_response
        results = await agent.delegate(
            tasks=["slow task"],
            config=SubagentConfig(timeout=0.01),
        )

    assert len(results) == 1
    assert results[0].status == "timeout"


@pytest.mark.asyncio
async def test_delegate_one_role_applies_role_system_prompt_to_every_task():
    provider = FakeCompletionProvider([
        LLMResponse(content="review a"),
        LLMResponse(content="review b"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider, system_prompt="Parent prompt"))

    results = await agent.delegate(
        tasks=["Task A", "Task B"],
        roles=[AgentRole(name="reviewer", system_prompt="Role prompt")],
    )

    assert [result.role for result in results] == ["reviewer", "reviewer"]
    assert [result.task for result in results] == ["Task A", "Task B"]
    assert all(result.status == "success" for result in results)
    for call in provider.calls:
        system_messages = [message["content"] for message in call["messages"] if message["role"] == "system"]
        assert system_messages[0] == "Parent prompt\n\nRole prompt"


@pytest.mark.asyncio
async def test_delegate_many_roles_with_one_task_fans_out_to_each_role():
    provider = FakeCompletionProvider([
        LLMResponse(content="review"),
        LLMResponse(content="security"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    results = await agent.delegate(
        tasks=["Inspect this"],
        roles=[
            AgentRole(name="reviewer", system_prompt="Review it"),
            AgentRole(name="security", system_prompt="Secure it"),
        ],
    )

    assert [result.role for result in results] == ["reviewer", "security"]
    assert [result.task for result in results] == ["Inspect this", "Inspect this"]
    assert provider.calls[0]["messages"][-1] == {"role": "user", "content": "Inspect this"}
    assert provider.calls[1]["messages"][-1] == {"role": "user", "content": "Inspect this"}


@pytest.mark.asyncio
async def test_delegate_rejects_role_task_count_mismatch():
    provider = FakeCompletionProvider([])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    with pytest.raises(ValueError, match="roles and tasks"):
        await agent.delegate(
            tasks=["A", "B", "C"],
            roles=[AgentRole(name="one"), AgentRole(name="two")],
        )


@pytest.mark.asyncio
async def test_delegate_does_not_pollute_parent_conversation_history():
    provider = FakeCompletionProvider([LLMResponse(content="done")])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    results = await agent.delegate(["Task A"], roles=[AgentRole(name="worker")])

    assert results[0].status == "success"
    assert agent._conversations == {}


@pytest.mark.asyncio
async def test_delegate_role_memory_scope_is_isolated():
    provider = FakeCompletionProvider([
        LLMResponse(content="alpha done"),
        LLMResponse(content="beta done"),
    ])
    memory = InMemoryMemoryStore([
        MemoryRecord(id="a", scope="conversation:alpha", kind="fact", content="alpha-only fact"),
        MemoryRecord(id="b", scope="conversation:beta", kind="fact", content="beta-only fact"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider, memory=memory, memory_enabled=True))

    results = await agent.delegate(
        tasks=["Use scoped fact"],
        roles=[
            AgentRole(name="alpha", memory_scope="alpha"),
            AgentRole(name="beta", memory_scope="beta"),
        ],
    )

    assert [result.role for result in results] == ["alpha", "beta"]
    first_context = "\n".join(str(message.get("content", "")) for message in provider.calls[0]["messages"])
    second_context = "\n".join(str(message.get("content", "")) for message in provider.calls[1]["messages"])
    assert "alpha-only fact" in first_context
    assert "beta-only fact" not in first_context
    assert "beta-only fact" in second_context
    assert "alpha-only fact" not in second_context


@pytest.mark.asyncio
async def test_delegate_role_tools_are_child_local():
    provider = FakeCompletionProvider([
        LLMResponse(content="", tool_calls=[LLMToolCall(id="tc_1", name="role_add", arguments='{"a": 1, "b": 2}')]),
        LLMResponse(content="sum is 3"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    async def role_add(a: int, b: int) -> int:
        return a + b

    results = await agent.delegate(
        tasks=["Use the role tool"],
        roles=[AgentRole(name="calculator", tools=[role_add])],
    )

    assert results[0].content == "sum is 3"
    assert any(tool["function"]["name"] == "role_add" for tool in provider.calls[0]["tools"])
    assert not agent._registry.has_tool("role_add")


@pytest.mark.asyncio
async def test_delegate_captures_child_events_and_emits_parent_subagent_events():
    parent_events = []
    provider = FakeCompletionProvider([LLMResponse(content="done")])
    agent = Agent(AgentConfig(
        model="fake-model",
        provider=provider,
        enable_tracing=True,
        event_handlers=[parent_events.append],
    ))

    results = await agent.delegate(["Trace me"], roles=[AgentRole(name="worker")])

    assert [event.type for event in results[0].events] == ["llm_start", "llm_end", "done"]
    parent_event_types = [event.type for event in parent_events]
    assert "subagent_start" in parent_event_types
    assert "subagent_end" in parent_event_types


@pytest.mark.asyncio
async def test_delegate_multiple_failures_return_structured_results_without_aborting_siblings():
    provider = FakeCompletionProvider([
        RuntimeError("first exploded"),
        LLMResponse(content="second ok"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    results = await agent.delegate(
        tasks=["fail first", "succeed second"],
        roles=[AgentRole(name="first"), AgentRole(name="second")],
    )

    assert [result.status for result in results] == ["error", "success"]
    assert results[0].role == "first"
    assert results[0].task == "fail first"
    assert "first exploded" in results[0].content
    assert results[1].content == "second ok"


@pytest.mark.asyncio
async def test_delegate_concat_aggregation_returns_single_structured_result():
    provider = FakeCompletionProvider([
        LLMResponse(content="alpha output"),
        LLMResponse(content="beta output"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    result = await agent.delegate(
        tasks=["A", "B"],
        roles=[AgentRole(name="alpha"), AgentRole(name="beta")],
        aggregation="concat",
    )

    assert isinstance(result, SubagentResult)
    assert result.status == "success"
    assert "[0] role=alpha status=success task=A" in result.content
    assert "alpha output" in result.content
    assert "[1] role=beta status=success task=B" in result.content
    assert result.metadata == {"aggregation": "concat", "count": 2}


@pytest.mark.asyncio
async def test_delegate_summarize_aggregation_calls_provider_once_without_tools():
    provider = FakeCompletionProvider([
        LLMResponse(content="alpha output"),
        LLMResponse(content="beta output"),
        LLMResponse(content="summary output"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    result = await agent.delegate(
        tasks=["A", "B"],
        roles=[AgentRole(name="alpha"), AgentRole(name="beta")],
        aggregation="summarize",
    )

    assert result.status == "success"
    assert result.content == "summary output"
    assert result.metadata["aggregation"] == "summarize"
    assert len(provider.calls) == 3
    assert provider.calls[-1]["tools"] is None
    summary_prompt = provider.calls[-1]["messages"][-1]["content"]
    assert "alpha output" in summary_prompt
    assert "beta output" in summary_prompt


@pytest.mark.asyncio
async def test_delegate_vote_aggregation_returns_winner_with_reason_metadata():
    provider = FakeCompletionProvider([
        LLMResponse(content="candidate 0"),
        LLMResponse(content="candidate 1"),
        LLMResponse(content='{"winner_index": 1, "reason": "more complete"}'),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    result = await agent.delegate(
        tasks=["A", "B"],
        roles=[AgentRole(name="alpha"), AgentRole(name="beta")],
        aggregation="vote",
    )

    assert result.status == "success"
    assert result.content == "candidate 1"
    assert result.role == "beta"
    assert result.task == "B"
    assert result.metadata["aggregation"] == "vote"
    assert result.metadata["winner_index"] == 1
    assert result.metadata["reason"] == "more complete"
    assert provider.calls[-1]["tools"] is None


@pytest.mark.asyncio
async def test_delegate_vote_aggregation_malformed_json_returns_controlled_error():
    provider = FakeCompletionProvider([
        LLMResponse(content="candidate 0"),
        LLMResponse(content="candidate 1"),
        LLMResponse(content="not json"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    result = await agent.delegate(
        tasks=["A", "B"],
        roles=[AgentRole(name="alpha"), AgentRole(name="beta")],
        aggregation="vote",
    )

    assert result.status == "error"
    assert "vote aggregation failed" in result.content
    assert result.metadata["aggregation"] == "vote"


@pytest.mark.asyncio
async def test_delegate_vote_aggregation_out_of_range_winner_returns_controlled_error():
    provider = FakeCompletionProvider([
        LLMResponse(content="candidate 0"),
        LLMResponse(content="candidate 1"),
        LLMResponse(content='{"winner_index": 99, "reason": "bad index"}'),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    result = await agent.delegate(
        tasks=["A", "B"],
        roles=[AgentRole(name="alpha"), AgentRole(name="beta")],
        aggregation="vote",
    )

    assert result.status == "error"
    assert "winner_index" in result.content
    assert result.metadata["aggregation"] == "vote"


@pytest.mark.asyncio
async def test_delegate_callable_aggregation_supports_sync_string_return():
    provider = FakeCompletionProvider([
        LLMResponse(content="alpha output"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    result = await agent.delegate(
        tasks=["A"],
        roles=[AgentRole(name="alpha")],
        aggregation=lambda results: f"custom: {results[0].content}",
    )

    assert result.status == "success"
    assert result.content == "custom: alpha output"
    assert result.metadata["aggregation"] == "callable"


@pytest.mark.asyncio
async def test_delegate_callable_aggregation_supports_async_subagent_result_return():
    provider = FakeCompletionProvider([
        LLMResponse(content="alpha output"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    async def aggregate(results):
        return SubagentResult(status="success", content=f"async: {results[0].content}")

    result = await agent.delegate(
        tasks=["A"],
        roles=[AgentRole(name="alpha")],
        aggregation=aggregate,
    )

    assert result.status == "success"
    assert result.content == "async: alpha output"
