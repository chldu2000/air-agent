from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from air_agent.providers import OpenAIProvider


def _mock_tool_call(*, id: str | None = None, name: str | None = None, arguments: str | None = None):
    tool_call = MagicMock()
    tool_call.id = id
    tool_call.function = MagicMock()
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


def _mock_response(*, content: str | None, tool_calls=None, usage=None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _mock_stream_chunk(*, content=None, tool_calls=None, usage=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls

    choice = MagicMock()
    choice.delta = delta

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _mock_stream_response(chunks) -> AsyncIterator:
    async def _aiter():
        for chunk in chunks:
            yield chunk

    stream = MagicMock()
    stream.__aiter__ = lambda self=None: _aiter()
    return stream


@pytest.mark.asyncio
async def test_complete_maps_text_usage_tool_calls_and_kwargs():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    usage = MagicMock(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    client.chat.completions.create.return_value = _mock_response(
        content="hello",
        tool_calls=[
            _mock_tool_call(id="call_1", name="lookup", arguments='{"query":"hi"}'),
            _mock_tool_call(id="call_2", name=None, arguments=None),
        ],
        usage=usage,
    )

    provider = OpenAIProvider(client=client)
    response = await provider.complete(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=0.2,
        max_tokens=64,
    )

    assert response.content == "hello"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 18
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == '{"query":"hi"}'
    assert response.tool_calls[1].id == "call_2"
    assert response.tool_calls[1].name == ""
    assert response.tool_calls[1].arguments == ""

    client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=0.2,
        max_tokens=64,
    )


@pytest.mark.asyncio
async def test_complete_maps_none_content_to_empty_string():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_mock_response(content=None, usage=None)
    )

    provider = OpenAIProvider(client=client)
    response = await provider.complete(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.content == ""
    assert response.usage is None
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_complete_returns_none_usage_for_partial_usage_payload():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_mock_response(
            content="hello",
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=None, total_tokens=9),
        )
    )

    provider = OpenAIProvider(client=client)
    response = await provider.complete(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.content == "hello"
    assert response.usage is None


@pytest.mark.asyncio
async def test_stream_maps_deltas_and_usage_and_omits_empty_tools():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value = _mock_stream_response(
        [
            _mock_stream_chunk(content="Hel"),
            _mock_stream_chunk(
                tool_calls=[
                    _mock_tool_call(id="call_1", name="lookup", arguments='{"q":"hi"}'),
                    _mock_tool_call(id=None, name=None, arguments=None),
                ]
            ),
            _mock_stream_chunk(content="lo"),
            _mock_stream_chunk(usage=MagicMock(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
        ]
    )

    provider = OpenAIProvider(client=client)
    chunks = []
    async for chunk in provider.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        temperature=0.3,
        stream=False,
        stream_options={},
    ):
        chunks.append(chunk)

    assert [chunk.content_delta for chunk in chunks] == ["Hel", "", "lo", ""]
    assert chunks[1].tool_call_deltas[0].index == 0
    assert chunks[1].tool_call_deltas[0].id == "call_1"
    assert chunks[1].tool_call_deltas[0].name == "lookup"
    assert chunks[1].tool_call_deltas[0].arguments == '{"q":"hi"}'
    assert chunks[1].tool_call_deltas[1].id is None
    assert chunks[1].tool_call_deltas[1].name is None
    assert chunks[1].tool_call_deltas[1].arguments == ""
    assert chunks[-1].usage.prompt_tokens == 2
    assert chunks[-1].usage.completion_tokens == 3
    assert chunks[-1].usage.total_tokens == 5

    client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.3,
    )
