from __future__ import annotations

import asyncio
import inspect
import json
import time
from json import JSONDecodeError
from typing import Any, Callable, Awaitable

from air_agent.types import ToolExecutionResult
from air_agent.tools.base import Tool


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


class _ToolRaisedTimeoutError(Exception):
    def __init__(self, original: TimeoutError) -> None:
        super().__init__(str(original))
        self.original = original


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    origin = getattr(annotation, "__origin__", None)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list or origin is list:
        return {"type": "array"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {}


def _extract_parameters(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        schema = _python_type_to_json_schema(param.annotation)
        if not schema and param.annotation is not inspect.Parameter.empty:
            schema = {"type": "string"}
        if not schema:
            schema = {"type": "string"}
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        func: Callable,
        name: str | None = None,
        description: str = "",
        *,
        conflict: str = "replace",
    ) -> None:
        tool_name = name or func.__name__
        if conflict == "error" and tool_name in self._tools:
            raise ValueError(f"Tool already registered: {tool_name}")
        tool_desc = description or func.__doc__ or ""
        parameters = _extract_parameters(func)
        self._tools[tool_name] = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
            is_mcp=False,
        )

    def register_mcp_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict], Awaitable[Any]],
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            is_mcp=True,
        )

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments_json: str) -> str:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        tool = self._tools[name]
        args = json.loads(arguments_json)
        if tool.is_mcp:
            result = await tool.handler(args)
        else:
            result = await tool.handler(**args)
        return str(result)

    async def execute_with_result(
        self,
        name: str,
        arguments_json: str,
        *,
        timeout: float | None = None,
    ) -> ToolExecutionResult:
        start = time.perf_counter()
        if name not in self._tools:
            return ToolExecutionResult.failure(
                content=f"Tool not found: {name}",
                error_kind="tool_not_found",
                duration_ms=_elapsed_ms(start),
            )

        tool = self._tools[name]
        try:
            args = json.loads(arguments_json)
            if not isinstance(args, dict):
                return ToolExecutionResult.failure(
                    content=f"Invalid arguments for tool '{name}': arguments must be a JSON object",
                    error_kind="invalid_arguments",
                    duration_ms=_elapsed_ms(start),
                )
            if not tool.is_mcp:
                try:
                    inspect.signature(tool.handler).bind(**args)
                except TypeError as exc:
                    return ToolExecutionResult.failure(
                        content=f"Invalid arguments for tool '{name}': {exc}",
                        error_kind="invalid_arguments",
                        duration_ms=_elapsed_ms(start),
                    )

            async def call_tool() -> Any:
                try:
                    if tool.is_mcp:
                        return await tool.handler(args)
                    return await tool.handler(**args)
                except TimeoutError as exc:
                    raise _ToolRaisedTimeoutError(exc) from exc

            if timeout is not None:
                try:
                    result = await asyncio.wait_for(call_tool(), timeout=timeout)
                except _ToolRaisedTimeoutError as exc:
                    return ToolExecutionResult.failure(
                        content=f"Error executing tool '{name}': {exc}",
                        error_kind="tool_error",
                        duration_ms=_elapsed_ms(start),
                    )
                except asyncio.TimeoutError:
                    return ToolExecutionResult.failure(
                        content=f"Tool timed out after {timeout}s: {name}",
                        error_kind="timeout",
                        duration_ms=_elapsed_ms(start),
                    )
            else:
                result = await call_tool()
        except JSONDecodeError as exc:
            return ToolExecutionResult.failure(
                content=f"Invalid JSON arguments for tool '{name}': {exc}",
                error_kind="invalid_arguments",
                duration_ms=_elapsed_ms(start),
            )
        except PermissionError as exc:
            return ToolExecutionResult.failure(
                content=f"Permission denied executing tool '{name}': {exc}",
                error_kind="permission_denied",
                duration_ms=_elapsed_ms(start),
            )
        except _ToolRaisedTimeoutError as exc:
            return ToolExecutionResult.failure(
                content=f"Error executing tool '{name}': {exc}",
                error_kind="tool_error",
                duration_ms=_elapsed_ms(start),
            )
        except Exception as exc:
            return ToolExecutionResult.failure(
                content=f"Error executing tool '{name}': {exc}",
                error_kind="tool_error",
                duration_ms=_elapsed_ms(start),
            )

        return ToolExecutionResult.success(
            content=str(result),
            duration_ms=_elapsed_ms(start),
        )

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def clone(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry._tools = dict(self._tools)
        return registry
