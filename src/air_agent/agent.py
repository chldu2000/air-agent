from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from air_agent.config import AgentConfig, SubagentConfig
from air_agent.mcp.client import MCPClient
from air_agent.tools.builtin import register_builtin_tools
from air_agent.tools.builtin.config import BuiltinToolsConfig
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import LLMSkillRouter
from air_agent.subagent import delegate as _delegate
from air_agent.tools.registry import ToolRegistry
from air_agent.types import Response, StreamEvent, SubagentResult, TokenUsage

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.default_headers,
        )
        self._registry = ToolRegistry()
        builtin_cfg = config.builtin_tools or BuiltinToolsConfig()
        register_builtin_tools(self._registry, builtin_cfg)
        self._mcp_clients: list[MCPClient] = []
        self._conversations: dict[str, list[dict[str, Any]]] = {}
        self._skill_manager: SkillManager | None = None
        self._skill_router: LLMSkillRouter | None = None
        if config.skills_dir:
            self._skill_manager = SkillManager(config.skills_dir)
            self._skill_manager.load()
            self._skill_router = LLMSkillRouter(client=self._client, model=config.model)

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
        if stream:
            return await self._run_stream(messages, conversation_id)
        return await self._run(messages, conversation_id)

    async def _run(self, messages: list[dict], conversation_id: str | None) -> Response:
        tools = self._registry.get_openai_tools() or None
        history: list[dict[str, Any]] = list(messages)

        if self._skill_manager and self._skill_manager.skills:
            matched = await self._skill_router.match(
                user_input=messages[-1]["content"],
                skills=self._skill_manager.skills,
            )
            for skill in matched:
                header = f'<skill name="{skill.name}" path="{skill.skill_dir}">\n'
                header += f"{skill.content}\n</skill>"
                history.insert(0, {
                    "role": "system",
                    "content": header,
                })

        for _ in range(self.config.max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": history,
            }
            if tools:
                kwargs["tools"] = tools

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message
            history.append(_message_to_dict(assistant_msg))

            if not assistant_msg.tool_calls:
                usage = None
                if response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                result = Response(content=assistant_msg.content or "", usage=usage, history=history)
                if conversation_id:
                    self._conversations[conversation_id] = history[-20:]
                return result

            results = await _execute_tool_calls(self._registry, assistant_msg.tool_calls)
            for tc, tool_result in zip(assistant_msg.tool_calls, results):
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        return Response(content="Reached maximum iterations without completion.", history=history)

    async def _run_stream(
        self, messages: list[dict], conversation_id: str | None
    ) -> AsyncIterator[StreamEvent]:
        tools = self._registry.get_openai_tools() or None
        history: list[dict[str, Any]] = list(messages)

        if self._skill_manager and self._skill_manager.skills:
            matched = await self._skill_router.match(
                user_input=messages[-1]["content"],
                skills=self._skill_manager.skills,
            )
            for skill in matched:
                header = f'<skill name="{skill.name}" path="{skill.skill_dir}">\n'
                header += f"{skill.content}\n</skill>"
                history.insert(0, {
                    "role": "system",
                    "content": header,
                })

        async def _stream_generator():
            for iteration in range(self.config.max_iterations):
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": history,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools

                stream = await self._client.chat.completions.create(**kwargs)

                text_content = ""
                tool_calls_map: dict[int, dict[str, Any]] = {}
                usage_data = None

                async for chunk in stream:
                    if chunk.usage:
                        usage_data = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    if delta.content:
                        text_content += delta.content
                        yield StreamEvent(type="text", content=delta.content)

                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": tc_chunk.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_chunk.id:
                                tool_calls_map[idx]["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    tool_calls_map[idx]["name"] = tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    tool_calls_map[idx]["arguments"] += tc_chunk.function.arguments

                if not tool_calls_map:
                    assistant_msg = {"role": "assistant", "content": text_content}
                    history.append(assistant_msg)
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
                    try:
                        result = await self._registry.execute(tc["name"], tc["arguments"])
                        results.append(result)
                        yield StreamEvent(type="tool_result", name=tc["name"], content=result)
                    except Exception as e:
                        err_msg = f"Error: {e}"
                        results.append(err_msg)
                        yield StreamEvent(type="tool_result", name=tc["name"], content=err_msg)

                for tc, tool_result in zip(tool_calls_list, results):
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

            yield StreamEvent(type="done", usage=None)

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


async def _execute_tool_calls(registry: ToolRegistry, tool_calls: list) -> list[str]:
    async def _exec_one(tc):
        try:
            return await registry.execute(tc.function.name, tc.function.arguments)
        except Exception as e:
            return f"Error executing tool '{tc.function.name}': {e}"

    return await asyncio.gather(*[_exec_one(tc) for tc in tool_calls])
