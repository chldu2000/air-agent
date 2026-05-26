from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuiltinToolsConfig:
    """Configuration for built-in tools."""

    enabled: bool = True
    tools: list[str] | None = None
    allowed_directories: list[str] = field(default_factory=list)
    max_read_size: int = 1_000_000
    default_timeout: float = 30.0
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "mkfs",
        "dd if=",
        "sudo",
        ":(){ :|:& };:",
        "shutdown",
        "reboot",
        "init 0",
        "init 6",
        "> /dev/sd",
        "chmod -R 777 /",
    ])
    max_find_results: int = 200
    max_grep_results: int = 100
    max_list_entries: int = 500
    max_output_bytes: int = 50_000

    def get_allowed_paths(self) -> list[Path]:
        raw = self.allowed_directories or [os.getcwd()]
        return [Path(p).resolve() for p in raw]

    @classmethod
    def from_dict(cls, data: dict) -> BuiltinToolsConfig:
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})
