from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from air_agent import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamToolCallDelta,
    LLMToolCall,
    OpenAIProvider,
)
from air_agent.providers import (
    LLMProvider as ProvidersLLMProvider,
    LLMResponse as ProvidersLLMResponse,
    LLMStreamChunk as ProvidersLLMStreamChunk,
    LLMStreamToolCallDelta as ProvidersLLMStreamToolCallDelta,
    LLMToolCall as ProvidersLLMToolCall,
    OpenAIProvider as ProvidersOpenAIProvider,
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
    assert ProvidersLLMToolCall is LLMToolCall
    assert ProvidersLLMResponse is LLMResponse
    assert ProvidersLLMStreamToolCallDelta is LLMStreamToolCallDelta
    assert ProvidersLLMStreamChunk is LLMStreamChunk
    assert ProvidersLLMProvider is LLMProvider
    assert ProvidersOpenAIProvider is OpenAIProvider


class FakeProvider:
    supports_tools = True
    supports_streaming = True

    async def complete(self, *, model, messages, tools=None, **options):
        assert model == "test-model"
        assert messages == [{"role": "user", "content": "hello"}]
        assert tools == [{"type": "function", "function": {"name": "lookup"}}]
        assert options == {"temperature": 0.2}
        return LLMResponse(
            content="done",
            tool_calls=[LLMToolCall(id="call_1", name="lookup", arguments="{}")],
        )

    async def stream(self, *, model, messages, tools=None, **options) -> AsyncIterator[LLMStreamChunk]:
        assert model == "test-model"
        assert messages == [{"role": "user", "content": "hello"}]
        assert tools is None
        assert options == {"temperature": 0.2}
        yield LLMStreamChunk(content_delta="d")
        yield LLMStreamChunk(
            tool_call_deltas=[
                LLMStreamToolCallDelta(index=0, id="call_1", name="lookup", arguments="{}")
            ]
        )


@pytest.mark.asyncio
async def test_provider_protocol_can_be_assigned_and_used():
    provider: LLMProvider = FakeProvider()
    assert provider.supports_tools is True
    assert provider.supports_streaming is True

    response = await provider.complete(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=0.2,
    )
    assert response.content == "done"
    assert response.tool_calls[0].name == "lookup"

    chunks = []
    async for chunk in provider.stream(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
    ):
        chunks.append(chunk)

    assert chunks[0].content_delta == "d"
    assert chunks[1].tool_call_deltas[0].name == "lookup"
