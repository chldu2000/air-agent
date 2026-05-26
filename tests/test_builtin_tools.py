from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from air_agent.tools.builtin.config import BuiltinToolsConfig
from air_agent.tools.builtin._permissions import (
    PermissionDeniedError,
    resolve_and_check_path,
    check_shell_command,
)
from air_agent.tools.builtin import register_builtin_tools
from air_agent.tools.registry import ToolRegistry
from air_agent.agent import Agent
from air_agent.config import AgentConfig


class TestBuiltinToolsConfig:
    def test_defaults(self):
        cfg = BuiltinToolsConfig()
        assert cfg.enabled is True
        assert cfg.tools is None
        assert cfg.allowed_directories == []
        assert cfg.max_read_size == 1_000_000
        assert cfg.max_find_results == 200
        assert cfg.max_grep_results == 100
        assert cfg.max_list_entries == 500
        assert cfg.max_output_bytes == 50_000

    def test_from_dict(self):
        cfg = BuiltinToolsConfig.from_dict({
            "enabled": False,
            "max_read_size": 500,
            "unknown_field": "ignored",
        })
        assert cfg.enabled is False
        assert cfg.max_read_size == 500

    def test_get_allowed_paths_cwd_fallback(self):
        cfg = BuiltinToolsConfig()
        paths = cfg.get_allowed_paths()
        assert len(paths) == 1
        assert paths[0] == Path.cwd()

    def test_custom_allowed_dirs(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        paths = cfg.get_allowed_paths()
        assert paths == [tmp_path.resolve()]


class TestPermissions:
    def test_path_inside_sandbox(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        result = resolve_and_check_path(str(tmp_path / "test.txt"), cfg)
        assert result == (tmp_path / "test.txt").resolve()

    def test_path_outside_sandbox_raises(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        with pytest.raises(PermissionDeniedError, match="outside allowed"):
            resolve_and_check_path("/etc/passwd", cfg)

    def test_path_traversal_blocked(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        with pytest.raises(PermissionDeniedError):
            resolve_and_check_path(str(tmp_path / ".." / ".." / "etc" / "passwd"), cfg)

    def test_must_exist_flag(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        with pytest.raises(FileNotFoundError):
            resolve_and_check_path(str(tmp_path / "nonexistent.txt"), cfg, must_exist=True)

    def test_shell_blocked_command(self):
        cfg = BuiltinToolsConfig()
        with pytest.raises(PermissionDeniedError, match="blocked"):
            check_shell_command("rm -rf /", cfg)

    def test_shell_blocked_sudo(self):
        cfg = BuiltinToolsConfig()
        with pytest.raises(PermissionDeniedError, match="blocked"):
            check_shell_command("sudo apt install something", cfg)

    def test_shell_allowed_command(self):
        cfg = BuiltinToolsConfig()
        check_shell_command("echo hello", cfg)  # no error


class TestRegisterBuiltinTools:
    def test_all_tools_registered(self):
        registry = ToolRegistry()
        cfg = BuiltinToolsConfig(allowed_directories=["."])
        register_builtin_tools(registry, cfg)
        for name in ["read_file", "write_file", "list_directory", "find_files", "grep", "run_shell"]:
            assert registry.has_tool(name)

    def test_select_tools(self):
        registry = ToolRegistry()
        cfg = BuiltinToolsConfig(tools=["read_file", "grep"], allowed_directories=["."])
        register_builtin_tools(registry, cfg)
        assert registry.has_tool("read_file")
        assert registry.has_tool("grep")
        assert not registry.has_tool("write_file")
        assert not registry.has_tool("run_shell")

    def test_disabled(self):
        registry = ToolRegistry()
        cfg = BuiltinToolsConfig(enabled=False)
        register_builtin_tools(registry, cfg)
        assert len(registry._tools) == 0


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_basic(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "test.txt").write_text("hello world")
        result = await registry.execute("read_file", json.dumps({"path": str(tmp_path / "test.txt")}))
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "lines.txt").write_text("line1\nline2\nline3\nline4\nline5\n")
        result = await registry.execute("read_file", json.dumps({
            "path": str(tmp_path / "lines.txt"),
            "offset": 1,
            "limit": 2,
        }))
        assert result == "line2\nline3\n"

    @pytest.mark.asyncio
    async def test_read_outside_sandbox(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        with pytest.raises(PermissionDeniedError, match="outside allowed"):
            await registry.execute("read_file", json.dumps({"path": "/etc/passwd"}))

    @pytest.mark.asyncio
    async def test_read_truncation(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)], max_read_size=10)
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "big.txt").write_text("a" * 100)
        result = await registry.execute("read_file", json.dumps({"path": str(tmp_path / "big.txt")}))
        assert "[TRUNCATED" in result


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_basic(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        result = await registry.execute("write_file", json.dumps({
            "path": str(tmp_path / "out.txt"),
            "content": "hello",
        }))
        assert "OK" in result
        assert (tmp_path / "out.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        result = await registry.execute("write_file", json.dumps({
            "path": str(tmp_path / "sub" / "dir" / "file.txt"),
            "content": "nested",
        }))
        assert "OK" in result
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"


class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_basic(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "file.txt").write_text("hi")
        (tmp_path / "subdir").mkdir()
        result = await registry.execute("list_directory", json.dumps({"path": str(tmp_path)}))
        assert "DIR  subdir/" in result
        assert "FILE file.txt" in result


class TestFindFiles:
    @pytest.mark.asyncio
    async def test_find_basic(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await registry.execute("find_files", json.dumps({
            "pattern": "*.py",
            "directory": str(tmp_path),
        }))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result


class TestGrep:
    @pytest.mark.asyncio
    async def test_grep_basic(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "code.py").write_text("def hello():\n    print('hello world')\n")
        result = await registry.execute("grep", json.dumps({
            "pattern": "hello",
            "path": str(tmp_path),
            "include": "*.py",
        }))
        assert "code.py" in result
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path: Path):
        cfg = BuiltinToolsConfig(allowed_directories=[str(tmp_path)])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        (tmp_path / "code.py").write_text("def foo(): pass\n")
        result = await registry.execute("grep", json.dumps({
            "pattern": "nonexistent",
            "path": str(tmp_path),
        }))
        assert "No matches found" in result


class TestRunShell:
    @pytest.mark.asyncio
    async def test_echo(self):
        cfg = BuiltinToolsConfig(allowed_directories=["."])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        result = await registry.execute("run_shell", json.dumps({"command": "echo hello"}))
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        cfg = BuiltinToolsConfig(allowed_directories=["."])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        with pytest.raises(PermissionDeniedError, match="blocked"):
            await registry.execute("run_shell", json.dumps({"command": "sudo rm -rf /"}))

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        cfg = BuiltinToolsConfig(allowed_directories=["."])
        registry = ToolRegistry()
        register_builtin_tools(registry, cfg)
        result = await registry.execute("run_shell", json.dumps({"command": "exit 1"}))
        assert "Exit code: 1" in result


class TestAgentBuiltinIntegration:
    def test_builtin_tools_auto_registered(self):
        agent = Agent(AgentConfig(model="gpt-4o", api_key="test"))
        for name in ["read_file", "write_file", "list_directory", "find_files", "grep", "run_shell"]:
            assert agent._registry.has_tool(name)

    def test_builtin_tools_disabled(self):
        agent = Agent(AgentConfig(
            model="gpt-4o",
            api_key="test",
            builtin_tools=BuiltinToolsConfig(enabled=False),
        ))
        for name in ["read_file", "write_file", "run_shell"]:
            assert not agent._registry.has_tool(name)

    def test_builtin_tools_coexist_with_custom(self):
        agent = Agent(AgentConfig(model="gpt-4o", api_key="test"))

        @agent.tool(name="custom", description="Custom tool")
        async def custom(x: int) -> int:
            return x

        assert agent._registry.has_tool("custom")
        assert agent._registry.has_tool("read_file")

    def test_builtin_config_from_json(self, tmp_path: Path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model": "gpt-4o",
            "builtin_tools": {"enabled": False},
        }))
        config = AgentConfig.from_json(str(config_file))
        assert config.builtin_tools is not None
        assert config.builtin_tools.enabled is False

    def test_builtin_config_from_env(self, monkeypatch):
        monkeypatch.setenv("AIR_BUILTIN_TOOLS", json.dumps({"enabled": False}))
        config = AgentConfig.from_env()
        assert config.builtin_tools is not None
        assert config.builtin_tools.enabled is False
