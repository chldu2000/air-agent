from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from air_agent.skills.skill import Skill, parse_skill_file

logger = logging.getLogger(__name__)


class SkillManager:
    def __init__(self, skills_dir: str | Path | list[str | Path]) -> None:
        if isinstance(skills_dir, list):
            self.skills_dirs = [Path(path) for path in skills_dir]
        else:
            self.skills_dirs = [Path(skills_dir)]
        self.skills_dir = self.skills_dirs[0]
        self.skills: list[Skill] = []

    def load(self) -> None:
        """Scan skills_dir for subdirectories containing SKILL.md files."""
        self.skills.clear()

        for skills_dir in self.skills_dirs:
            if not skills_dir.is_dir():
                logger.warning("Skills directory does not exist: %s", skills_dir)
                continue

            for entry in sorted(skills_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                if not skill_file.is_file():
                    logger.warning("Skipping directory without SKILL.md: %s", entry)
                    continue
                skill = parse_skill_file(skill_file)
                if skill is None:
                    logger.warning("Skipping invalid SKILL.md: %s", skill_file)
                    continue
                self.skills.append(skill)

        logger.info("Loaded %d skill(s) from %s", len(self.skills), self.skills_dirs)

    def metadata_summary(self) -> str:
        """Generate compact summary of all skills for system prompt injection."""
        if not self.skills:
            return ""
        lines = [f"- {s.name}: {s.description}" for s in self.skills]
        return "\n".join(lines)

    def get_skill(self, name: str) -> Skill | None:
        """Get skill by exact name match."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def list_attachments(self, skill: Skill, *, max_entries: int = 100) -> dict[str, Any]:
        """Return a bounded manifest of files bundled with a skill."""
        attachments: list[dict[str, Any]] = []
        all_files: list[Path] = []
        for path in sorted(skill.skill_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(skill.skill_dir)
            if path == skill.path or any(part.startswith(".") for part in relative.parts):
                continue
            all_files.append(path)

        for path in all_files[:max_entries]:
            relative = path.relative_to(skill.skill_dir).as_posix()
            attachments.append({
                "path": relative,
                "type": "file",
                "size": path.stat().st_size,
            })

        return {
            "attachments": attachments,
            "count": len(all_files),
            "truncated": len(all_files) > max_entries,
        }

    def render_skill_payload(self, name: str, description: str | None = None) -> str:
        """Render full skill instructions and attachment metadata for use_skill."""
        skill = self.get_skill(name)
        if skill is None:
            available = ", ".join(skill.name for skill in self.skills) or "(none)"
            return f"Skill not found: {name}\nAvailable skills: {available}"

        attachment_info = self.list_attachments(skill)
        lines = [
            f"# Skill: {skill.name}",
            f"Description: {skill.description}",
            f"Path: {skill.skill_dir}",
        ]
        if description:
            lines.append(f"Requested context: {description}")
        lines.extend(["", "## Instructions", skill.content or "(empty)", "", "## Attachments"])

        attachments = attachment_info["attachments"]
        if not attachments:
            lines.append("None")
        else:
            for attachment in attachments:
                lines.append(
                    f"- {attachment['path']} ({attachment['type']}, {attachment['size']} bytes)"
                )
        if attachment_info["truncated"]:
            lines.append(
                f"- ... truncated after {len(attachments)} of {attachment_info['count']} attachment(s)"
            )
        return "\n".join(lines)
