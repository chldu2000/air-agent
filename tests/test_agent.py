import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from air_agent.agent import Agent
from air_agent.config import AgentConfig
from air_agent.types import Response


def _mock_openai_response(content: str, tool_calls=None, usage=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=usage.get("prompt_tokens", 10) if usage else 10,
        completion_tokens=usage.get("completion_tokens", 20) if usage else 20,
        total_tokens=usage.get("total_tokens", 30) if usage else 30,
    )
    return resp


@pytest.mark.asyncio
async def test_basic_conversation():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    mock_response = _mock_openai_response("Hello! How can I help?")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        result = await agent.run("Hi")

    assert isinstance(result, Response)
    assert result.content == "Hello! How can I help?"
    assert result.usage.total_tokens == 30
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_system_prompt_in_messages():
    config = AgentConfig(model="gpt-4o", api_key="test-key", system_prompt="You are helpful")
    agent = Agent(config)

    mock_response = _mock_openai_response("Hi!")
    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        await agent.run("Hello")

    call_kwargs = mock_create.call_args
    messages = call_kwargs.kwargs["messages"] if call_kwargs.kwargs else call_kwargs[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"


@pytest.mark.asyncio
async def test_max_iterations_prevents_infinite_loop():
    config = AgentConfig(model="gpt-4o", api_key="test-key", max_iterations=1)
    agent = Agent(config)

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "add"
    tool_call.function.arguments = '{"a": 1, "b": 2}'

    mock_response = _mock_openai_response(None, tool_calls=[tool_call])

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        result = await agent.run("calculate")

    assert "reached maximum" in result.content.lower() or result.content != ""


@pytest.mark.asyncio
async def test_tool_calling_loop():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    async def add(a: int, b: int) -> int:
        return a + b

    agent.add_tools([add])

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "add"
    tool_call.function.arguments = '{"a": 3, "b": 5}'

    resp1 = _mock_openai_response(None, tool_calls=[tool_call])
    resp2 = _mock_openai_response("The result is 8.")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [resp1, resp2]
        result = await agent.run("What is 3+5?")

    assert result.content == "The result is 8."
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _mock_openai_response("First")
        await agent.run("Turn 1", conversation_id="s1")

        mock_create.return_value = _mock_openai_response("Second")
        result = await agent.run("Turn 2", conversation_id="s1")

    assert result.content == "Second"
    # Second call should include conversation history
    second_call_msgs = mock_create.call_args.kwargs["messages"]
    assert len(second_call_msgs) > 1


@pytest.mark.asyncio
async def test_decorator_tool_registration():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    @agent.tool(name="greet", description="Greet someone")
    async def greet(name: str) -> str:
        return f"Hello, {name}"

    assert agent._registry.has_tool("greet")


@pytest.mark.asyncio
async def test_run_emits_llm_and_done_events_when_tracing_enabled():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _mock_openai_response("Hello")
        result = await agent.run("Hi", conversation_id="conv_1")

    assert result.content == "Hello"
    assert [event.type for event in events] == ["llm_start", "llm_end", "done"]
    assert events[0].run_id == events[1].run_id == events[2].run_id
    assert events[0].conversation_id == "conv_1"
    assert events[0].iteration == 0
    assert events[1].usage.total_tokens == 30
    assert events[2].content == "Hello"


@pytest.mark.asyncio
async def test_run_emits_tool_start_and_tool_end_events():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    async def add(a: int, b: int) -> int:
        return a + b

    agent.add_tools([add])

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "add"
    tool_call.function.arguments = '{"a": 3, "b": 5}'

    resp1 = _mock_openai_response(None, tool_calls=[tool_call])
    resp2 = _mock_openai_response("The result is 8.")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [resp1, resp2]
        result = await agent.run("What is 3+5?")

    assert result.content == "The result is 8."
    event_types = [event.type for event in events]
    assert event_types == [
        "llm_start",
        "llm_end",
        "tool_start",
        "tool_end",
        "llm_start",
        "llm_end",
        "done",
    ]
    tool_start = events[2]
    tool_end = events[3]
    assert tool_start.name == "add"
    assert tool_start.arguments == '{"a": 3, "b": 5}'
    assert tool_end.name == "add"
    assert tool_end.content == "8"
    assert tool_end.duration_ms is not None


@pytest.mark.asyncio
async def test_run_emits_tool_error_event_with_error_kind():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "missing"
    tool_call.function.arguments = "{}"

    resp1 = _mock_openai_response(None, tool_calls=[tool_call])
    resp2 = _mock_openai_response("I could not call the tool.")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [resp1, resp2]
        await agent.run("Use missing tool")

    tool_error = [event for event in events if event.type == "tool_error"][0]
    assert tool_error.name == "missing"
    assert tool_error.error_kind == "tool_not_found"
    assert "Tool not found: missing" in tool_error.content


@pytest.mark.asyncio
async def test_run_does_not_emit_events_when_tracing_disabled():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=False,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _mock_openai_response("Hello")
        result = await agent.run("Hi")

    assert result.content == "Hello"
    assert events == []


@pytest.mark.asyncio
async def test_tool_retry_event_emitted_before_successful_retry():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
        max_tool_retries=1,
    )
    agent = Agent(config)
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return "ok"

    agent.tool(name="flaky", description="Flaky tool")(flaky)

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "flaky"
    tool_call.function.arguments = "{}"

    resp1 = _mock_openai_response(None, tool_calls=[tool_call])
    resp2 = _mock_openai_response("Recovered.")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [resp1, resp2]
        result = await agent.run("Call flaky")

    assert result.content == "Recovered."
    assert calls == 2
    retry_event = [event for event in events if event.type == "retry"][0]
    assert retry_event.name == "flaky"
    assert retry_event.error_kind == "tool_error"
    assert retry_event.attempt == 1
    tool_end = [event for event in events if event.type == "tool_end"][0]
    assert tool_end.content == "ok"
    assert tool_end.attempt == 1


@pytest.mark.asyncio
async def test_invalid_arguments_not_retried_when_tool_retries_enabled():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
        max_tool_retries=2,
    )
    agent = Agent(config)
    calls = 0

    async def needs_value(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    agent.tool(name="needs_value", description="Needs value")(needs_value)

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "needs_value"
    tool_call.function.arguments = "{}"

    result = await agent._execute_tool_call_with_events(
        tool_call,
        run_id="run_1",
        conversation_id=None,
        iteration=0,
    )

    tool_starts = [event for event in events if event.type == "tool_start"]
    tool_errors = [event for event in events if event.type == "tool_error"]
    retries = [event for event in events if event.type == "retry"]
    assert calls == 0
    assert len(tool_starts) == 1
    assert tool_starts[0].attempt == 0
    assert len(tool_errors) == 1
    assert tool_errors[0].error_kind == "invalid_arguments"
    assert tool_errors[0].attempt == 0
    assert retries == []
    assert "Invalid arguments" in result


@pytest.mark.asyncio
async def test_tool_error_retried_until_last_failure_returned():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
        max_tool_retries=2,
    )
    agent = Agent(config)
    calls = 0

    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"boom {calls}")

    agent.tool(name="always_fails", description="Always fails")(always_fails)

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "always_fails"
    tool_call.function.arguments = "{}"

    result = await agent._execute_tool_call_with_events(
        tool_call,
        run_id="run_1",
        conversation_id=None,
        iteration=0,
    )

    assert calls == 3
    assert "boom 3" in result
    assert [event.attempt for event in events if event.type == "tool_start"] == [0, 1, 2]
    assert [event.attempt for event in events if event.type == "tool_error"] == [0, 1, 2]
    assert [event.attempt for event in events if event.type == "retry"] == [1, 2]
