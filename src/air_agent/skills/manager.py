from __future__ import annotations

import logging
from pathlib import Path

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
