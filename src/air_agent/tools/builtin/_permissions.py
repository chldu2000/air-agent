from __future__ import annotations

from pathlib import Path

from air_agent.tools.builtin.config import BuiltinToolsConfig


class PermissionDeniedError(PermissionError):
    """Raised when a tool action violates sandbox restrictions."""


def resolve_and_check_path(
    path: str,
    config: BuiltinToolsConfig,
    *,
    must_exist: bool = False,
) -> Path:
    resolved = Path(path).resolve()
    allowed = config.get_allowed_paths()

    if not any(_is_under(resolved, base) for base in allowed):
        raise PermissionDeniedError(
            f"Path '{path}' is outside allowed directories: "
            f"{[str(p) for p in allowed]}"
        )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return resolved


def check_shell_command(
    command: str,
    config: BuiltinToolsConfig,
) -> None:
    lower_cmd = command.lower().strip()
    for blocked in config.blocked_commands:
        if blocked.lower() in lower_cmd:
            raise PermissionDeniedError(
                f"Command blocked by security policy: matched pattern '{blocked}'"
            )


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
