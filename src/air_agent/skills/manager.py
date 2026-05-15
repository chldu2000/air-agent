from __future__ import annotations

import logging
from pathlib import Path

from air_agent.skills.skill import Skill, parse_skill_file

logger = logging.getLogger(__name__)


class SkillManager:
    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills: list[Skill] = []

    def load(self) -> None:
        """Scan skills_dir for .md files and parse them."""
        self.skills.clear()

        if not self.skills_dir.is_dir():
            logger.warning("Skills directory does not exist: %s", self.skills_dir)
            return

        for path in sorted(self.skills_dir.glob("*.md")):
            skill = parse_skill_file(path)
            if skill is None:
                logger.warning("Skipping invalid skill file: %s", path)
                continue
            self.skills.append(skill)

        logger.info("Loaded %d skill(s) from %s", len(self.skills), self.skills_dir)

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