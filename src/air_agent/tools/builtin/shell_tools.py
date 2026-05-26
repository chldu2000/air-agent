from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from air_agent.tools.builtin.config import BuiltinToolsConfig
from air_agent.tools.builtin._permissions import check_shell_command


def make_shell_tools(
    config: BuiltinToolsConfig,
) -> list[tuple[Callable[..., Awaitable[Any]], str, str]]:
    async def run_shell(command: str, timeout: float | None = None) -> str:
        """Execute a shell command and return its output."""
        check_shell_command(command, config)
        effective_timeout = min(
            timeout if timeout is not None else config.default_timeout,
            config.default_timeout,
        )
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"[TIMEOUT: command exceeded {effective_timeout}s]"
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        truncated = False
        if len(output.encode("utf-8")) > config.max_output_bytes:
            output = output[: config.max_output_bytes]
            truncated = True
        result = output.strip()
        if proc.returncode != 0:
            result += f"\n[Exit code: {proc.returncode}]"
        if truncated:
            result += (
                f"\n\n[TRUNCATED: output exceeded {config.max_output_bytes} bytes."
                f" Pipe to head/tail or redirect to a file.]"
            )
        return result

    return [
        (run_shell, "run_shell", "Execute a shell command and return its output."),
    ]
