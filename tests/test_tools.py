import asyncio

import pytest
from air_agent.tools.registry import ToolRegistry


def test_register_and_get_openai_tools():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")
    tools = registry.get_openai_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "add"
    assert tools[0]["function"]["description"] == "Add two numbers"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "a" in tools[0]["function"]["parameters"]["properties"]
    assert "b" in tools[0]["function"]["parameters"]["properties"]


def test_register_detects_required_params():
    registry = ToolRegistry()

    async def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}"

    registry.register(greet, name="greet", description="Greet someone")
    params = registry.get_openai_tools()[0]["function"]["parameters"]
    assert "name" in params["required"]
    assert "greeting" not in params["required"]


@pytest.mark.asyncio
async def test_execute_local_tool():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")
    result = await registry.execute("add", '{"a": 3, "b": 5}')
    assert result == "8"


@pytest.mark.asyncio
async def test_execute_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="unknown"):
        await registry.execute("unknown", "{}")


def test_register_mcp_tool():
    registry = ToolRegistry()

    async def handler(arguments: dict) -> str:
        return "result"

    registry.register_mcp_tool(
        name="mcp_read",
        description="Read via MCP",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )
    tools = registry.get_openai_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "mcp_read"


@pytest.mark.asyncio
async def test_execute_mcp_tool():
    registry = ToolRegistry()

    async def handler(arguments: dict) -> str:
        return f"read {arguments['path']}"

    registry.register_mcp_tool(
        name="mcp_read",
        description="Read via MCP",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )
    result = await registry.execute("mcp_read", '{"path": "/tmp/a"}')
    assert result == "read /tmp/a"


def test_has_tool():
    registry = ToolRegistry()

    async def noop() -> str:
        return ""

    registry.register(noop, name="noop", description="Does nothing")
    assert registry.has_tool("noop")
    assert not registry.has_tool("missing")


@pytest.mark.asyncio
async def test_execute_with_result_success():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")

    result = await registry.execute_with_result("add", '{"a": 3, "b": 5}')

    assert result.ok is True
    assert result.content == "8"
    assert result.error_kind is None
    assert result.duration_ms is not None


@pytest.mark.asyncio
async def test_execute_with_result_invalid_arguments():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")

    result = await registry.execute_with_result("add", "{not json")

    assert result.ok is False
    assert result.error_kind == "invalid_arguments"
    assert "Invalid JSON arguments" in result.content


@pytest.mark.asyncio
async def test_execute_with_result_rejects_non_object_arguments():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")

    result = await registry.execute_with_result("add", "[]")

    assert result.ok is False
    assert result.error_kind == "invalid_arguments"
    assert "arguments must be a JSON object" in result.content


@pytest.mark.asyncio
async def test_execute_with_result_missing_required_argument():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")

    result = await registry.execute_with_result("add", "{}")

    assert result.ok is False
    assert result.error_kind == "invalid_arguments"
    assert "Invalid arguments for tool 'add'" in result.content


@pytest.mark.asyncio
async def test_execute_with_result_tool_not_found():
    registry = ToolRegistry()

    result = await registry.execute_with_result("missing", "{}")

    assert result.ok is False
    assert result.error_kind == "tool_not_found"
    assert "Tool not found: missing" in result.content


@pytest.mark.asyncio
async def test_execute_with_result_timeout():
    registry = ToolRegistry()

    async def slow() -> str:
        await asyncio.sleep(1)
        return "late"

    registry.register(slow, name="slow", description="Slow tool")

    result = await registry.execute_with_result("slow", "{}", timeout=0.01)

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert "Tool timed out after 0.01s: slow" in result.content


@pytest.mark.asyncio
async def test_execute_with_result_tool_raised_timeout_without_wrapper_timeout():
    registry = ToolRegistry()

    async def raises_timeout() -> str:
        raise TimeoutError("inner")

    registry.register(raises_timeout, name="raises_timeout", description="Raises timeout")

    result = await registry.execute_with_result("raises_timeout", "{}", timeout=None)

    assert result.ok is False
    assert result.error_kind == "tool_error"
    assert result.content == "Error executing tool 'raises_timeout': inner"


@pytest.mark.asyncio
async def test_execute_with_result_permission_denied():
    registry = ToolRegistry()

    async def blocked() -> str:
        raise PermissionError("denied")

    registry.register(blocked, name="blocked", description="Blocked tool")

    result = await registry.execute_with_result("blocked", "{}")

    assert result.ok is False
    assert result.error_kind == "permission_denied"
    assert "Permission denied executing tool 'blocked': denied" in result.content


@pytest.mark.asyncio
async def test_execute_preserves_existing_raise_behavior_for_unknown_tool():
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="missing"):
        await registry.execute("missing", "{}")
