import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, AsyncIterator
from air_agent.agent import Agent
from air_agent.config import AgentConfig
from air_agent.providers import LLMStreamChunk, LLMStreamToolCallDelta
from air_agent.types import TokenUsage


class FakeStreamingProvider:
    supports_tools = True
    supports_streaming = True

    def __init__(self, batches: list[list[LLMStreamChunk]]) -> None:
        self.batches = list(batches)
        self.stream_calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        raise AssertionError("complete should not be used for streaming")

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        self.stream_calls.append(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "tools": [dict(tool) for tool in tools] if tools else tools,
                "options": dict(options),
            }
        )
        for chunk in self.batches.pop(0):
            yield chunk


def _mock_stream_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.role = "assistant"

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    if usage:
        chunk.usage = MagicMock(**usage)
    return chunk


def _mock_stream_response(chunks):
    resp = MagicMock()

    async def _aiter():
        for c in chunks:
            yield c

    resp.__aiter__ = lambda self: _aiter()
    return resp


@pytest.mark.asyncio
async def test_streaming_uses_custom_provider_for_text():
    provider = FakeStreamingProvider(
        [[
            LLMStreamChunk(content_delta="Hello"),
            LLMStreamChunk(
                content_delta=" world",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            ),
        ]]
    )
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    stream_gen = await agent.run("Hi", stream=True)
    events = []
    async for event in stream_gen:
        events.append(event)

    assert [event.type for event in events] == ["text", "text", "done"]
    assert events[2].usage is not None
    assert events[2].usage.total_tokens == 3
    assert provider.stream_calls[0]["model"] == "fake-model"


@pytest.mark.asyncio
async def test_streaming_uses_custom_provider_for_tool_call():
    provider = FakeStreamingProvider(
        [
            [
                LLMStreamChunk(
                    tool_call_deltas=[
                        LLMStreamToolCallDelta(index=0, id="tc_1", name="add", arguments='{"a": 2'),
                    ]
                ),
                LLMStreamChunk(
                    tool_call_deltas=[
                        LLMStreamToolCallDelta(index=0, arguments=', "b": 4}'),
                    ]
                ),
            ],
            [LLMStreamChunk(content_delta="The result is 6.")],
        ]
    )
    agent = Agent(AgentConfig(model="fake-model", provider=provider))

    @agent.tool()
    async def add(a: int, b: int) -> int:
        return a + b

    stream_gen = await agent.run("What is 2+4?", stream=True)
    events = []
    async for event in stream_gen:
        events.append(event)

    assert [event.type for event in events] == ["tool_call", "tool_result", "text", "done"]
    assert events[1].content == "6"
    assert len(provider.stream_calls) == 2
    assert provider.stream_calls[1]["messages"][-1]["role"] == "tool"
    assert provider.stream_calls[1]["messages"][-1]["tool_call_id"] == "tc_1"


@pytest.mark.asyncio
async def test_streaming_text():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    chunks = [
        _mock_stream_chunk(content="Hello"),
        _mock_stream_chunk(content=" world"),
        _mock_stream_chunk(content="!", finish_reason="stop"),
        _mock_stream_chunk(usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}),
    ]
    stream_resp = _mock_stream_response(chunks)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = stream_resp
        stream_gen = await agent.run("Hi", stream=True)

        events = []
        async for event in stream_gen:
            events.append(event)

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 3
    assert text_events[0].content == "Hello"
    assert text_events[1].content == " world"
    assert text_events[2].content == "!"

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_streaming_emits_tracing_events_for_text_response():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    chunks = [
        _mock_stream_chunk(content="Hello"),
        _mock_stream_chunk(content=" world", finish_reason="stop"),
        _mock_stream_chunk(usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
    ]
    stream_resp = _mock_stream_response(chunks)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = stream_resp
        stream_gen = await agent.run("Hi", stream=True)

        stream_events = []
        async for event in stream_gen:
            stream_events.append(event)

    assert [event.type for event in stream_events] == ["text", "text", "done"]
    assert mock_create.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert [event.type for event in events] == ["llm_start", "llm_end", "done"]
    assert events[1].usage.total_tokens == 7
    assert events[2].content == "Hello world"


@pytest.mark.asyncio
async def test_streaming_emits_tool_error_tracing_event():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=True,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    function = MagicMock()
    function.name = "missing"
    function.arguments = "{}"
    tool_delta = MagicMock()
    tool_delta.index = 0
    tool_delta.id = "tc_1"
    tool_delta.function = function

    chunks = [
        _mock_stream_chunk(tool_calls=[tool_delta]),
    ]
    stream_resp = _mock_stream_response(chunks)
    final_resp = _mock_stream_response([
        _mock_stream_chunk(content="Handled error.", finish_reason="stop"),
    ])

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [stream_resp, final_resp]
        stream_gen = await agent.run("Call missing", stream=True)

        stream_events = []
        async for event in stream_gen:
            stream_events.append(event)

    assert any(event.type == "tool_result" for event in stream_events)
    tool_error = [event for event in events if event.type == "tool_error"][0]
    assert tool_error.name == "missing"
    assert tool_error.error_kind == "tool_not_found"


@pytest.mark.asyncio
async def test_streaming_emits_tool_start_and_tool_end_tracing_events():
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

    function = MagicMock()
    function.name = "add"
    function.arguments = '{"a": 2, "b": 4}'
    tool_delta = MagicMock()
    tool_delta.index = 0
    tool_delta.id = "tc_1"
    tool_delta.function = function

    stream_resp = _mock_stream_response([
        _mock_stream_chunk(tool_calls=[tool_delta]),
    ])
    final_resp = _mock_stream_response([
        _mock_stream_chunk(content="The result is 6.", finish_reason="stop"),
    ])

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [stream_resp, final_resp]
        stream_gen = await agent.run("What is 2+4?", stream=True)

        stream_events = []
        async for event in stream_gen:
            stream_events.append(event)

    assert [event.type for event in stream_events] == [
        "tool_call",
        "tool_result",
        "text",
        "done",
    ]
    tool_start = [event for event in events if event.type == "tool_start"][0]
    tool_end = [event for event in events if event.type == "tool_end"][0]
    assert tool_start.name == "add"
    assert tool_start.arguments == '{"a": 2, "b": 4}'
    assert tool_end.name == "add"
    assert tool_end.content == "6"


@pytest.mark.asyncio
async def test_streaming_does_not_emit_tracing_events_when_disabled():
    events = []
    config = AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        enable_tracing=False,
        log_events=False,
        event_handlers=[events.append],
    )
    agent = Agent(config)

    stream_resp = _mock_stream_response([
        _mock_stream_chunk(content="Hello", finish_reason="stop"),
    ])

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = stream_resp
        stream_gen = await agent.run("Hi", stream=True)

        stream_events = []
        async for event in stream_gen:
            stream_events.append(event)

    assert [event.type for event in stream_events] == ["text", "done"]
    assert events == []
