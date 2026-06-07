from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI

from air_agent.providers.types import LLMResponse, LLMStreamChunk, LLMStreamToolCallDelta, LLMToolCall
from air_agent.types import TokenUsage


class OpenAIProvider:
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        if client is None:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
            )
        self.client = client

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **options,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) or ""
        return LLMResponse(
            content=content,
            tool_calls=_tool_calls_from_message(message),
            usage=_usage_from_object(getattr(response, "usage", None)),
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **options,
        }
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        async for chunk in stream:
            usage = _usage_from_object(getattr(chunk, "usage", None))
            delta = getattr(chunk.choices[0], "delta", None) if getattr(chunk, "choices", None) else None
            content_delta = getattr(delta, "content", None) or ""
            tool_call_deltas = _stream_tool_call_deltas_from_delta(delta)

            if content_delta or tool_call_deltas or usage is not None:
                yield LLMStreamChunk(
                    content_delta=content_delta,
                    tool_call_deltas=tool_call_deltas,
                    usage=usage,
                )


def _usage_from_object(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if not all(
        isinstance(value, int)
        for value in (prompt_tokens, completion_tokens, total_tokens)
    ):
        return None
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _tool_calls_from_message(message: Any) -> list[LLMToolCall]:
    tool_calls = getattr(message, "tool_calls", None) or []
    return [
        LLMToolCall(
            id=str(getattr(tool_call, "id", "") or ""),
            name=str(getattr(getattr(tool_call, "function", None), "name", "") or ""),
            arguments=str(getattr(getattr(tool_call, "function", None), "arguments", "") or ""),
        )
        for tool_call in tool_calls
    ]


def _stream_tool_call_deltas_from_delta(delta: Any) -> list[LLMStreamToolCallDelta]:
    tool_calls = getattr(delta, "tool_calls", None) or []
    deltas: list[LLMStreamToolCallDelta] = []
    for fallback_index, tool_call in enumerate(tool_calls):
        index = getattr(tool_call, "index", None)
        if not isinstance(index, int):
            index = fallback_index
        deltas.append(
            LLMStreamToolCallDelta(
                index=cast(int, index),
                id=getattr(tool_call, "id", None),
                name=getattr(getattr(tool_call, "function", None), "name", None),
                arguments=str(
                    getattr(getattr(tool_call, "function", None), "arguments", "") or ""
                ),
            )
        )
    return deltas
