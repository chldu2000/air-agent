from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from air_agent import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamToolCallDelta,
    LLMToolCall,
    TokenUsage,
)
from air_agent.providers import (
    LLMProvider as ProvidersLLMProvider,
    LLMResponse as ProvidersLLMResponse,
    LLMStreamChunk as ProvidersLLMStreamChunk,
    LLMStreamToolCallDelta as ProvidersLLMStreamToolCallDelta,
    LLMToolCall as ProvidersLLMToolCall,
    TokenUsage as ProvidersTokenUsage,
)


def test_provider_types_default_shapes():
    tool_call = LLMToolCall(id="call_1", name="lookup", arguments='{"query":"hi"}')
    assert tool_call.id == "call_1"
    assert tool_call.name == "lookup"
    assert tool_call.arguments == '{"query":"hi"}'

    response = LLMResponse(content="hello")
    assert response.content == "hello"
    assert response.tool_calls == []
    assert response.usage is None

    delta = LLMStreamToolCallDelta(index=2)
    assert delta.index == 2
    assert delta.id is None
    assert delta.name is None
    assert delta.arguments == ""

    chunk = LLMStreamChunk()
    assert chunk.content_delta == ""
    assert chunk.tool_call_deltas == []
    assert chunk.usage is None


def test_provider_package_reexports_match_top_level():
    assert ProvidersTokenUsage is TokenUsage
    assert ProvidersLLMToolCall is LLMToolCall
    assert ProvidersLLMResponse is LLMResponse
    assert ProvidersLLMStreamToolCallDelta is LLMStreamToolCallDelta
    assert ProvidersLLMStreamChunk is LLMStreamChunk
    assert ProvidersLLMProvider is LLMProvider


class FakeProvider:
    supports_tools = True
    supports_streaming = True

    async def complete(self, messages, tools=None):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tools == [{"type": "function", "function": {"name": "lookup"}}]
        return LLMResponse(
            content="done",
            tool_calls=[LLMToolCall(id="call_1", name="lookup", arguments="{}")],
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def stream(self, messages, tools=None) -> AsyncIterator[LLMStreamChunk]:
        assert messages == [{"role": "user", "content": "hello"}]
        assert tools is None
        yield LLMStreamChunk(content_delta="d")
        yield LLMStreamChunk(
            tool_call_deltas=[
                LLMStreamToolCallDelta(index=0, id="call_1", name="lookup", arguments="{}")
            ]
        )


@pytest.mark.asyncio
async def test_provider_protocol_can_be_assigned_and_used():
    provider: LLMProvider = FakeProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.supports_tools is True
    assert provider.supports_streaming is True

    response = await provider.complete(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    assert response.content == "done"
    assert response.tool_calls[0].name == "lookup"
    assert response.usage.total_tokens == 3

    chunks = []
    async for chunk in provider.stream(messages=[{"role": "user", "content": "hello"}]):
        chunks.append(chunk)

    assert chunks[0].content_delta == "d"
    assert chunks[1].tool_call_deltas[0].name == "lookup"
