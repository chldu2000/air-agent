from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator
from uuid import uuid4

from air_agent.config import AgentConfig, SubagentConfig
from air_agent.mcp.client import MCPClient
from air_agent.providers import LLMToolCall, OpenAIProvider
from air_agent.tools.builtin import register_builtin_tools
from air_agent.tools.builtin.config import BuiltinToolsConfig
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import LLMSkillRouter, SkillRouteResult
from air_agent.subagent import delegate as _delegate
from air_agent.tools.registry import ToolRegistry
from air_agent.tracing import EventDispatcher
from air_agent.types import Response, RunEvent, StreamEvent, SubagentResult, TokenUsage

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._provider = _build_provider(config)
        self._client = getattr(self._provider, "client", None)
        self._registry = ToolRegistry()
        builtin_cfg = config.builtin_tools or BuiltinToolsConfig()
        register_builtin_tools(self._registry, builtin_cfg)
        self._mcp_clients: list[MCPClient] = []
        self._conversations: dict[str, list[dict[str, Any]]] = {}
        self._skill_manager: SkillManager | None = None
        self._skill_router: LLMSkillRouter | None = None
        self._events = EventDispatcher(
            enabled=config.enable_tracing or config.log_events,
            handlers=config.event_handlers,
            log_events=config.log_events,
        )
        if config.skills_dir:
            self._skill_manager = SkillManager(config.skills_dir)
            self._skill_manager.load()
            self._skill_router = LLMSkillRouter(provider=self._provider, model=config.model)

    def tool(self, name: str | None = None, description: str = ""):
        def decorator(func):
            self._registry.register(func, name=name, description=description)
            return func
        return decorator

    def add_tools(self, funcs: list) -> None:
        for func in funcs:
            self._registry.register(func)

    async def _connect_mcp(self) -> None:
        for server_conf in self.config.mcp_servers:
            client = MCPClient(server_conf, timeout=self.config.tool_timeout)
            try:
                await client.connect()
                tools = await client.list_tools()
                for t in tools:
                    async def _make_handler(mcp_client: MCPClient, tool_name: str):
                        async def handler(args: dict) -> str:
                            return await mcp_client.call_tool(tool_name, args)
                        return handler

                    handler = await _make_handler(client, t["name"])
                    self._registry.register_mcp_tool(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["inputSchema"],
                        handler=handler,
                    )
                self._mcp_clients.append(client)
            except Exception:
                logger.warning("Skipping MCP server %s: connection failed", server_conf, exc_info=True)

    async def _disconnect_mcp(self) -> None:
        for client in self._mcp_clients:
            await client.disconnect()
        self._mcp_clients.clear()

    async def __aenter__(self):
        await self._connect_mcp()
        return self

    async def __aexit__(self, *exc):
        await self._disconnect_mcp()

    async def _emit(self, event_type: str, **kwargs: Any) -> None:
        await self._events.emit(RunEvent(type=event_type, **kwargs))

    def _build_messages(self, user_input: str, conversation_id: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.config.system_prompt or self._skill_manager:
            system_content = self.config.system_prompt or ""
            if self._skill_manager:
                summary = self._skill_manager.metadata_summary()
                if summary:
                    system_content += f"\n\n## Available Skills\n{summary}"
            messages.append({"role": "system", "content": system_content})
        if conversation_id and conversation_id in self._conversations:
            messages.extend(self._conversations[conversation_id])
        messages.append({"role": "user", "content": user_input})
        return messages

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        stream: bool = False,
    ) -> Response | AsyncIterator[StreamEvent]:
        messages = self._build_messages(message, conversation_id)
        run_id = f"run_{uuid4().hex}"
        if stream:
            return await self._run_stream(messages, conversation_id, run_id)
        return await self._run(messages, conversation_id, run_id=run_id)

    async def _run(
        self,
        messages: list[dict],
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> Response:
        run_id = run_id or f"run_{uuid4().hex}"
        tools = self._registry.get_openai_tools() or None
        history: list[dict[str, Any]] = list(messages)

        history = await self._route_and_inject_skills(
            history,
            user_input=messages[-1]["content"],
            run_id=run_id,
            conversation_id=conversation_id,
        )

        for iteration in range(self.config.max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": history,
            }
            if tools:
                kwargs["tools"] = tools

            await self._emit(
                "llm_start",
                run_id=run_id,
                conversation_id=conversation_id,
                iteration=iteration,
                metadata={"tools_count": len(tools or [])},
            )
            llm_start = time.perf_counter()
            response = await self._provider.complete(**kwargs)
            llm_duration_ms = round((time.perf_counter() - llm_start) * 1000, 3)
            history.append(_provider_message_to_dict(response))

            usage = response.usage
            await self._emit(
                "llm_end",
                run_id=run_id,
                conversation_id=conversation_id,
                iteration=iteration,
                duration_ms=llm_duration_ms,
                usage=usage,
            )

            if not response.tool_calls:
                content = response.content or ""
                await self._emit(
                    "done",
                    run_id=run_id,
                    conversation_id=conversation_id,
                    iteration=iteration,
                    content=content,
                    usage=usage,
                )
                result = Response(content=response.content or "", usage=usage, history=history)
                if conversation_id:
                    self._conversations[conversation_id] = history[-20:]
                return result

            results = await asyncio.gather(*[
                self._execute_tool_call_with_events(
                    _provider_tool_call_to_object(tc),
                    run_id=run_id,
                    conversation_id=conversation_id,
                    iteration=iteration,
                )
                for tc in response.tool_calls
            ])
            for tc, tool_result in zip(response.tool_calls, results):
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        content = "Reached maximum iterations without completion."
        await self._emit(
            "done",
            run_id=run_id,
            conversation_id=conversation_id,
            content=content,
        )
        return Response(content=content, history=history)

    async def _route_and_inject_skills(
        self,
        history: list[dict[str, Any]],
        *,
        user_input: str,
        run_id: str,
        conversation_id: str | None,
    ) -> list[dict[str, Any]]:
        if not self._skill_manager or not self._skill_manager.skills or not self._skill_router:
            return history
        skills = self._skill_manager.skills
        await self._emit(
            "skill_route_start",
            run_id=run_id,
            conversation_id=conversation_id,
            metadata={
                "candidate_names": [skill.name for skill in skills],
                "candidate_count": len(skills),
                "router": type(self._skill_router).__name__,
            },
        )
        route_start = time.perf_counter()
        try:
            result = await self._skill_router.route(user_input=user_input, skills=skills)
        except Exception as exc:
            logger.warning("Skill router route failed", exc_info=True)
            error_type = type(exc).__name__
            result = SkillRouteResult(
                duration_ms=round((time.perf_counter() - route_start) * 1000, 3),
                error_type=error_type,
                error_message=str(exc) or error_type,
            )

        if result.error_type is not None:
            await self._emit(
                "skill_route_error",
                run_id=run_id,
                conversation_id=conversation_id,
                content=result.error_message or "",
                duration_ms=result.duration_ms,
                metadata={
                    "error_type": result.error_type,
                    "fallback": "no_skills",
                },
            )
            return history

        await self._emit(
            "skill_route_end",
            run_id=run_id,
            conversation_id=conversation_id,
            content=result.raw_output,
            duration_ms=result.duration_ms,
            metadata={
                "matched_names": [skill.name for skill in result.matched_skills],
                "unrecognized_names": result.unrecognized_names,
            },
        )
        for skill in result.matched_skills:
            header = f'<skill name="{skill.name}" path="{skill.skill_dir}">\n'
            header += f"{skill.content}\n</skill>"
            history.insert(0, {
                "role": "system",
                "content": header,
            })
            await self._emit(
                "skill_injected",
                run_id=run_id,
                conversation_id=conversation_id,
                name=skill.name,
                metadata={
                    "path": str(skill.skill_dir),
                    "content_length": len(skill.content),
                },
            )
        return history

    async def _execute_tool_call_with_events(
        self,
        tool_call: Any,
        *,
        run_id: str,
        conversation_id: str | None,
        iteration: int,
    ) -> str:
        name = tool_call.function.name
        arguments = tool_call.function.arguments
        max_attempt = max(0, self.config.max_tool_retries)
        retryable_errors = {"timeout", "tool_error"}
        last_failure_content = ""

        for attempt in range(max_attempt + 1):
            await self._emit(
                "tool_start",
                run_id=run_id,
                conversation_id=conversation_id,
                iteration=iteration,
                name=name,
                arguments=arguments,
                attempt=attempt,
            )
            result = await self._registry.execute_with_result(
                name,
                arguments,
                timeout=self.config.tool_timeout,
            )
            if result.ok:
                await self._emit(
                    "tool_end",
                    run_id=run_id,
                    conversation_id=conversation_id,
                    iteration=iteration,
                    name=name,
                    arguments=arguments,
                    content=result.content,
                    duration_ms=result.duration_ms,
                    error_kind=result.error_kind,
                    attempt=attempt,
                )
                return result.content

            last_failure_content = result.content
            await self._emit(
                "tool_error",
                run_id=run_id,
                conversation_id=conversation_id,
                iteration=iteration,
                name=name,
                arguments=arguments,
                content=result.content,
                duration_ms=result.duration_ms,
                error_kind=result.error_kind,
                attempt=attempt,
            )
            if attempt < max_attempt and result.error_kind in retryable_errors:
                await self._emit(
                    "retry",
                    run_id=run_id,
                    conversation_id=conversation_id,
                    iteration=iteration,
                    name=name,
                    arguments=arguments,
                    content=result.content,
                    error_kind=result.error_kind,
                    attempt=attempt + 1,
                )
                continue
            return result.content

        return last_failure_content

    async def _run_stream(
        self, messages: list[dict], conversation_id: str | None, run_id: str
    ) -> AsyncIterator[StreamEvent]:
        if not getattr(self._provider, "supports_streaming", False):
            raise RuntimeError(
                "Streaming is not available for the configured provider because it "
                "does not support streaming."
            )
        tools = self._registry.get_openai_tools() or None
        history: list[dict[str, Any]] = list(messages)
        history = await self._route_and_inject_skills(
            history,
            user_input=messages[-1]["content"],
            run_id=run_id,
            conversation_id=conversation_id,
        )

        async def _stream_generator():
            for iteration in range(self.config.max_iterations):
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": history,
                }
                if tools:
                    kwargs["tools"] = tools

                await self._emit(
                    "llm_start",
                    run_id=run_id,
                    conversation_id=conversation_id,
                    iteration=iteration,
                    metadata={"tools_count": len(tools or []), "stream": True},
                )
                llm_start = time.perf_counter()
                stream = self._provider.stream(**kwargs)

                text_content = ""
                tool_calls_map: dict[int, dict[str, Any]] = {}
                usage_data = None

                async for chunk in stream:
                    if chunk.usage:
                        usage_data = chunk.usage

                    if chunk.content_delta:
                        text_content += chunk.content_delta
                        yield StreamEvent(type="text", content=chunk.content_delta)

                    if chunk.tool_call_deltas:
                        for tc_chunk in chunk.tool_call_deltas:
                            idx = tc_chunk.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": tc_chunk.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_chunk.id:
                                tool_calls_map[idx]["id"] = tc_chunk.id
                            if tc_chunk.name:
                                tool_calls_map[idx]["name"] = tc_chunk.name
                            if tc_chunk.arguments:
                                tool_calls_map[idx]["arguments"] += tc_chunk.arguments

                llm_duration_ms = round((time.perf_counter() - llm_start) * 1000, 3)
                await self._emit(
                    "llm_end",
                    run_id=run_id,
                    conversation_id=conversation_id,
                    iteration=iteration,
                    duration_ms=llm_duration_ms,
                    usage=usage_data,
                )

                if not tool_calls_map:
                    assistant_msg = {"role": "assistant", "content": text_content}
                    history.append(assistant_msg)
                    await self._emit(
                        "done",
                        run_id=run_id,
                        conversation_id=conversation_id,
                        iteration=iteration,
                        content=text_content,
                        usage=usage_data,
                    )
                    yield StreamEvent(type="done", usage=usage_data)
                    if conversation_id:
                        self._conversations[conversation_id] = history[-20:]
                    return

                tool_calls_list = [tool_calls_map[i] for i in sorted(tool_calls_map)]
                history.append({
                    "role": "assistant",
                    "content": text_content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls_list
                    ],
                })

                for tc in tool_calls_list:
                    yield StreamEvent(type="tool_call", name=tc["name"], arguments=tc["arguments"])

                results = []
                for tc in tool_calls_list:
                    result = await self._execute_tool_call_with_events(
                        _stream_tool_call_to_object(tc),
                        run_id=run_id,
                        conversation_id=conversation_id,
                        iteration=iteration,
                    )
                    results.append(result)
                    yield StreamEvent(type="tool_result", name=tc["name"], content=result)

                for tc, tool_result in zip(tool_calls_list, results):
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

            content = "Reached maximum iterations without completion."
            await self._emit(
                "done",
                run_id=run_id,
                conversation_id=conversation_id,
                content=content,
                usage=None,
            )
            yield StreamEvent(type="done", content=content, usage=None)

        return _stream_generator()

    async def delegate(
        self,
        tasks: list[str],
        config: SubagentConfig | None = None,
    ) -> list[SubagentResult]:
        return await _delegate(self, tasks, config)


def _message_to_dict(msg: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _usage_from_response(response: Any) -> TokenUsage | None:
    if not response.usage:
        return None
    return TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )


def _stream_tool_call_to_object(tc: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=tc["id"],
        function=SimpleNamespace(
            name=tc["name"],
            arguments=tc["arguments"],
        ),
    )


def _build_provider(config: AgentConfig) -> Any:
    provider = config.provider
    if provider is None or provider == "openai":
        return OpenAIProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.default_headers,
        )
    if isinstance(provider, str):
        raise ValueError(f"Unsupported provider: {provider}")
    return provider


def _provider_message_to_dict(response: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content or None,
    }
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in response.tool_calls
        ]
    return message


def _provider_tool_call_to_object(tc: LLMToolCall) -> Any:
    return SimpleNamespace(
        id=tc.id,
        function=SimpleNamespace(
            name=tc.name,
            arguments=tc.arguments,
        ),
    )
