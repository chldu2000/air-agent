from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: Path


def parse_skill_file(path: Path) -> Skill | None:
    """Parse a SKILL.md file with YAML frontmatter.

    Returns None if frontmatter is missing or required fields (name, description)
    are absent.
    """
    text = Path(path).read_text(encoding="utf-8")

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not match:
        return None

    frontmatter_text = match.group(1)
    body = match.group(2)

    name = None
    description = None
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[len("name:"):].strip()
        elif line.startswith("description:"):
            description = line[len("description:"):].strip()

    if not name or not description:
        return None

    return Skill(
        name=name,
        description=description,
        content=body.strip(),
        path=Path(path),
    )