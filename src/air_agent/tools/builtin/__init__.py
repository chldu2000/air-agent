from __future__ import annotations

from air_agent.tools.registry import ToolRegistry
from air_agent.tools.builtin.config import BuiltinToolsConfig

_ALL_TOOLS = [
    "read_file",
    "write_file",
    "list_directory",
    "find_files",
    "grep",
    "run_shell",
]


def register_builtin_tools(
    registry: ToolRegistry,
    config: BuiltinToolsConfig,
) -> None:
    if not config.enabled:
        return

    from air_agent.tools.builtin.file_tools import make_file_tools
    from air_agent.tools.builtin.shell_tools import make_shell_tools

    selected = set(config.tools) if config.tools else set(_ALL_TOOLS)

    for func, name, description in make_file_tools(config):
        if name in selected:
            registry.register(func, name=name, description=description)

    for func, name, description in make_shell_tools(config):
        if name in selected:
            registry.register(func, name=name, description=description)
