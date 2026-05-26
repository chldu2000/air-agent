from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Awaitable

from air_agent.tools.builtin.config import BuiltinToolsConfig
from air_agent.tools.builtin._permissions import resolve_and_check_path


def make_file_tools(
    config: BuiltinToolsConfig,
) -> list[tuple[Callable[..., Awaitable[Any]], str, str]]:
    async def read_file(path: str, offset: int = 0, limit: int = -1) -> str:
        """Read file contents. Use offset/limit for large files."""
        resolved = resolve_and_check_path(path, config, must_exist=True)
        size = resolved.stat().st_size
        if size > config.max_read_size:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            truncated = text[: config.max_read_size]
            return (
                truncated
                + f"\n\n[TRUNCATED: file size ({size}) exceeds max_read_size"
                f" ({config.max_read_size}). Use offset/limit to read more.]"
            )
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)
        selected = lines[offset:]
        if limit > 0:
            selected = selected[:limit]
        return "".join(selected)

    async def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if it does not exist."""
        resolved = Path(path).resolve()
        parent = resolved.parent
        allowed = config.get_allowed_paths()
        if not any(
            _is_under(parent, base) or _is_under(resolved, base) for base in allowed
        ):
            raise PermissionError(
                f"Path '{path}' is outside allowed directories: {[str(p) for p in allowed]}"
            )
        parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {path}"

    async def list_directory(path: str = ".") -> str:
        """List directory contents."""
        resolved = resolve_and_check_path(path, config, must_exist=True)
        entries = list(sorted(resolved.iterdir()))
        total = len(entries)
        capped = entries[: config.max_list_entries]
        lines = []
        for entry in capped:
            if entry.is_dir():
                lines.append(f"DIR  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"FILE {entry.name}  ({size} bytes)")
        result = "\n".join(lines)
        if total > config.max_list_entries:
            result += (
                f"\n\n[TRUNCATED: {total} entries, showing first"
                f" {config.max_list_entries}.]"
            )
        return result

    async def find_files(pattern: str, directory: str = ".") -> str:
        """Find files matching a glob pattern."""
        resolved = resolve_and_check_path(directory, config, must_exist=True)
        matches = list(resolved.rglob(pattern))
        total = len(matches)
        capped = matches[: config.max_find_results]
        lines = [str(m.relative_to(resolved)) for m in sorted(capped)]
        result = "\n".join(lines)
        if total > config.max_find_results:
            result += (
                f"\n\n[TRUNCATED: {total} matches found, showing first"
                f" {config.max_find_results}. Use a more specific pattern to narrow results.]"
            )
        return result

    async def grep(pattern: str, path: str = ".", include: str = "*") -> str:
        """Search file contents for a regex pattern."""
        resolved = resolve_and_check_path(path, config, must_exist=True)
        regex = re.compile(pattern)
        matches = []
        for root, _dirs, files in os.walk(resolved):
            root_path = Path(root)
            for fname in sorted(files):
                if not _glob_match(fname, include):
                    continue
                fpath = root_path / fname
                if not any(_is_under(fpath.resolve(), base) for base in config.get_allowed_paths()):
                    continue
                try:
                    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                except (OSError, UnicodeDecodeError):
                    continue
                rel = fpath.relative_to(resolved)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append(f"{rel}:{i}: {line.rstrip()}")
                        if len(matches) >= config.max_grep_results:
                            total = len(matches)
                            result = "\n".join(matches)
                            result += (
                                f"\n\n[TRUNCATED: {total}+ matches found, showing first"
                                f" {config.max_grep_results}. Use a more specific pattern or include filter.]"
                            )
                            return result
                if len(matches) >= config.max_grep_results:
                    break
            if len(matches) >= config.max_grep_results:
                break
        return "\n".join(matches) if matches else "No matches found."

    return [
        (read_file, "read_file", "Read file contents. Use offset/limit for large files."),
        (write_file, "write_file", "Write content to a file, creating it if it does not exist."),
        (list_directory, "list_directory", "List directory contents with file sizes."),
        (find_files, "find_files", "Find files matching a glob pattern."),
        (grep, "grep", "Search file contents for a regex pattern."),
    ]


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(name, pattern)
