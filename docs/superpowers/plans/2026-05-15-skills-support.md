# Skills Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Skills system that loads SKILL.md files from a directory and injects relevant skill instructions into the agent's context via progressive prompt injection.

**Architecture:** New `skills/` sub-package with three files — `skill.py` (dataclass + parser), `router.py` (pluggable matching strategy), `manager.py` (orchestration). Agent gains a `SkillManager` at init time when `skills_dir` is configured. On each `run()`, skill metadata is always present in the system prompt; full skill content is injected only for matched skills via an LLM routing call.

**Tech Stack:** Python stdlib (`pathlib`, `re` for parsing), existing `openai` client for LLM routing. No new dependencies.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/air_agent/skills/__init__.py` | Public exports for skills sub-package |
| Create | `src/air_agent/skills/skill.py` | `Skill` dataclass + `parse_skill_file()` parser |
| Create | `src/air_agent/skills/router.py` | `SkillRouter` ABC + `LLMSkillRouter` default |
| Create | `src/air_agent/skills/manager.py` | `SkillManager`: scanning, loading, routing, prompt generation |
| Modify | `src/air_agent/config.py` | Add `skills_dir` field to `AgentConfig` |
| Modify | `src/air_agent/agent.py` | Integrate `SkillManager` into init + `_build_messages` + `_run` |
| Modify | `src/air_agent/__init__.py` | Export new public types |
| Create | `tests/test_skills.py` | All tests for skills functionality |

---

### Task 1: Skill dataclass and SKILL.md parser

**Files:**
- Create: `src/air_agent/skills/__init__.py`
- Create: `src/air_agent/skills/skill.py`
- Create: `tests/test_skills.py`

- [ ] **Step 1: Write the failing tests for Skill parsing**

Create `tests/test_skills.py`:

```python
from __future__ import annotations

import pytest
from pathlib import Path

from air_agent.skills.skill import Skill, parse_skill_file


class TestParseSkillFile:
    def test_valid_skill_file(self, tmp_path: Path):
        skill_file = tmp_path / "brainstorming.md"
        skill_file.write_text(
            "---\n"
            "name: brainstorming\n"
            "description: Use when starting creative work\n"
            "---\n"
            "# Brainstorming\n"
            "\n"
            "Ask questions one at a time.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is not None
        assert skill.name == "brainstorming"
        assert skill.description == "Use when starting creative work"
        assert "Ask questions one at a time." in skill.content
        assert skill.path == skill_file

    def test_missing_frontmatter_returns_none(self, tmp_path: Path):
        skill_file = tmp_path / "no_frontmatter.md"
        skill_file.write_text("# Just a heading\n\nSome text.\n")
        skill = parse_skill_file(skill_file)
        assert skill is None

    def test_missing_name_returns_none(self, tmp_path: Path):
        skill_file = tmp_path / "no_name.md"
        skill_file.write_text(
            "---\n"
            "description: Use when something\n"
            "---\n"
            "Content here.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is None

    def test_missing_description_returns_none(self, tmp_path: Path):
        skill_file = tmp_path / "no_desc.md"
        skill_file.write_text(
            "---\n"
            "name: some-skill\n"
            "---\n"
            "Content here.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is None

    def test_empty_content(self, tmp_path: Path):
        skill_file = tmp_path / "empty_body.md"
        skill_file.write_text(
            "---\n"
            "name: empty\n"
            "description: An empty skill\n"
            "---\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is not None
        assert skill.name == "empty"
        assert skill.content == ""

    def test_multiline_content(self, tmp_path: Path):
        skill_file = tmp_path / "multi.md"
        skill_file.write_text(
            "---\n"
            "name: multi\n"
            "description: Multi-line content\n"
            "---\n"
            "# Title\n"
            "\n"
            "## Section\n"
            "\n"
            "- Item 1\n"
            "- Item 2\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is not None
        assert "## Section" in skill.content
        assert "- Item 1" in skill.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'air_agent.skills'`

- [ ] **Step 3: Create the skills package and implement Skill + parser**

Create `src/air_agent/skills/__init__.py`:

```python
"""Skills sub-package for air-agent."""

from air_agent.skills.skill import Skill, parse_skill_file
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import SkillRouter, LLMSkillRouter

__all__ = [
    "Skill",
    "parse_skill_file",
    "SkillManager",
    "SkillRouter",
    "LLMSkillRouter",
]
```

Create `src/air_agent/skills/skill.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/air_agent/skills/__init__.py src/air_agent/skills/skill.py tests/test_skills.py
git commit -m "feat: add Skill dataclass and SKILL.md parser"
```

---

### Task 2: SkillManager — directory scanning and metadata summary

**Files:**
- Create: `src/air_agent/skills/manager.py`
- Modify: `tests/test_skills.py` — append SkillManager tests

- [ ] **Step 1: Write the failing tests for SkillManager**

Append to `tests/test_skills.py`:

```python
from air_agent.skills.manager import SkillManager


class TestSkillManager:
    def _create_skill_file(self, directory: Path, filename: str, name: str, description: str, content: str = ""):
        path = directory / filename
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n{content}\n"
        )

    def test_load_scans_directory(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "brainstorming.md", "brainstorming", "Use when creating")
        self._create_skill_file(skills_dir, "debugging.md", "debugging", "Use when bugs")

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 2
        names = {s.name for s in manager.skills}
        assert "brainstorming" in names
        assert "debugging" in names

    def test_load_skips_invalid_files(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "valid.md", "valid", "Use when valid")
        (skills_dir / "invalid.md").write_text("# No frontmatter\n")

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 1
        assert manager.skills[0].name == "valid"

    def test_load_ignores_non_md_files(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "skill.md", "skill", "Use when needed")
        (skills_dir / "notes.txt").write_text("Not a skill file\n")

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 1

    def test_metadata_summary(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "a.md", "brainstorming", "Use when creating")
        self._create_skill_file(skills_dir, "b.md", "debugging", "Use when bugs found")

        manager = SkillManager(skills_dir)
        manager.load()

        summary = manager.metadata_summary()
        assert "- brainstorming: Use when creating" in summary
        assert "- debugging: Use when bugs found" in summary

    def test_get_skill_by_name(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "brainstorming.md", "brainstorming", "Use when creating")

        manager = SkillManager(skills_dir)
        manager.load()

        skill = manager.get_skill("brainstorming")
        assert skill is not None
        assert skill.name == "brainstorming"

        assert manager.get_skill("nonexistent") is None

    def test_empty_directory_loads_nothing(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 0
        assert manager.metadata_summary() == ""

    def test_nonexistent_directory_loads_nothing(self, tmp_path: Path):
        manager = SkillManager(tmp_path / "does_not_exist")
        manager.load()

        assert len(manager.skills) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py::TestSkillManager -v`
Expected: FAIL — `ImportError: cannot import name 'SkillManager'`

- [ ] **Step 3: Implement SkillManager**

Create `src/air_agent/skills/manager.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py -v`
Expected: All tests PASS (both TestParseSkillFile and TestSkillManager)

- [ ] **Step 5: Commit**

```bash
git add src/air_agent/skills/manager.py tests/test_skills.py
git commit -m "feat: add SkillManager with directory scanning and metadata summary"
```

---

### Task 3: SkillRouter — pluggable routing strategy

**Files:**
- Create: `src/air_agent/skills/router.py`
- Modify: `tests/test_skills.py` — append SkillRouter tests

- [ ] **Step 1: Write the failing tests for SkillRouter**

Append to `tests/test_skills.py`:

```python
from unittest.mock import AsyncMock, MagicMock

from air_agent.skills.router import SkillRouter, LLMSkillRouter


class TestLLMSkillRouter:
    def _make_skill(self, name: str, description: str) -> Skill:
        return Skill(
            name=name,
            description=description,
            content=f"# {name}\nInstructions for {name}",
            path=Path("/fake") / f"{name}.md",
        )

    @pytest.mark.asyncio
    async def test_match_returns_relevant_skills(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "brainstorming, debugging"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            self._make_skill("debugging", "Use when bugs"),
            self._make_skill("deploy", "Use when deploying"),
        ]

        result = await router.match("I need to brainstorm ideas", skills)
        names = {s.name for s in result}
        assert "brainstorming" in names
        assert "debugging" in names
        assert "deploy" not in names

    @pytest.mark.asyncio
    async def test_match_returns_empty_when_no_match(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "none"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("deploy", "Use when deploying")]

        result = await router.match("Tell me a joke", skills)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_match_returns_empty_on_llm_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("test", "Use when testing")]

        result = await router.match("test query", skills)
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py::TestLLMSkillRouter -v`
Expected: FAIL — `ImportError: cannot import name 'SkillRouter'`

- [ ] **Step 3: Implement SkillRouter and LLMSkillRouter**

Create `src/air_agent/skills/router.py`:

```python
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from air_agent.skills.skill import Skill

logger = logging.getLogger(__name__)

_ROUTING_SYSTEM_PROMPT = """\
You are a skill router. Given a user input and a list of available skills, \
return ONLY the names of skills that are relevant, separated by commas. \
If no skills are relevant, respond with "none". \
Do not include any other text or explanation."""


class SkillRouter(ABC):
    @abstractmethod
    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        """Match user input against available skills."""


class LLMSkillRouter(SkillRouter):
    """Default implementation: use a lightweight LLM call to select relevant skills."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        if not skills:
            return []

        skill_list = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        messages = [
            {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Available skills:\n{skill_list}\n\nUser input: {user_input}"},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip().lower()
        except Exception:
            logger.warning("Skill routing LLM call failed", exc_info=True)
            return []

        if content == "none" or not content:
            return []

        matched_names = {name.strip() for name in content.split(",")}
        return [s for s in skills if s.name in matched_names]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/air_agent/skills/router.py tests/test_skills.py
git commit -m "feat: add SkillRouter ABC and LLMSkillRouter default implementation"
```

---

### Task 4: Add skills_dir to AgentConfig

**Files:**
- Modify: `src/air_agent/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests for skills_dir config**

Append to `tests/test_config.py`:

```python
class TestSkillsDirConfig:
    def test_skills_dir_from_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model": "gpt-4o",
            "skills_dir": "/path/to/skills",
        }))
        config = AgentConfig.from_json(str(config_file))
        assert config.skills_dir == "/path/to/skills"

    def test_skills_dir_default_is_none(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"model": "gpt-4o"}))
        config = AgentConfig.from_json(str(config_file))
        assert config.skills_dir is None

    def test_skills_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("AIR_SKILLS_DIR", "/my/skills")
        config = AgentConfig.from_env()
        assert config.skills_dir == "/my/skills"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_config.py::TestSkillsDirConfig -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'skills_dir'`

- [ ] **Step 3: Add skills_dir field and env parsing to AgentConfig**

In `src/air_agent/config.py`, add `skills_dir` to `AgentConfig` dataclass at line 55 (before `__post_init__`):

```python
    skills_dir: str | None = None
```

Add `skills_dir` env var parsing in `from_env()` method. Add to `env_map` dict (after line 82):

```python
            f"{prefix}SKILLS_DIR": ("skills_dir", str),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_config.py -v`
Expected: All config tests PASS (old + new)

- [ ] **Step 5: Commit**

```bash
git add src/air_agent/config.py tests/test_config.py
git commit -m "feat: add skills_dir field to AgentConfig"
```

---

### Task 5: Integrate SkillManager into Agent

**Files:**
- Modify: `src/air_agent/agent.py`
- Modify: `tests/test_skills.py` — append integration tests

- [ ] **Step 1: Write the failing tests for Agent + Skills integration**

Append to `tests/test_skills.py`:

```python
from air_agent.agent import Agent
from air_agent.config import AgentConfig


class TestAgentSkillsIntegration:
    def _create_skill_file(self, directory: Path, filename: str, name: str, description: str, content: str = ""):
        path = directory / filename
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n{content}\n"
        )

    def test_agent_initializes_skill_manager(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "test.md", "test-skill", "Use when testing")

        config = AgentConfig(model="gpt-4o", api_key="test-key", skills_dir=str(skills_dir))
        agent = Agent(config)

        assert agent._skill_manager is not None
        assert len(agent._skill_manager.skills) == 1

    def test_agent_without_skills_dir_has_no_manager(self):
        config = AgentConfig(model="gpt-4o", api_key="test-key")
        agent = Agent(config)
        assert agent._skill_manager is None

    @pytest.mark.asyncio
    async def test_skill_metadata_injected_into_system_prompt(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(skills_dir, "a.md", "brainstorming", "Use when creating")
        self._create_skill_file(skills_dir, "b.md", "debugging", "Use when bugs")

        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            system_prompt="You are helpful.",
            skills_dir=str(skills_dir),
        )
        agent = Agent(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "done"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            # First call: skill routing; second call: actual response
            routing_response = MagicMock()
            routing_response.choices = [MagicMock()]
            routing_response.choices[0].message.content = "none"
            mock_create.side_effect = [routing_response, mock_response]

            result = await agent.run("hello")

        # The second call (actual response) should contain skill metadata in messages
        actual_call = mock_create.call_args_list[1]
        messages = actual_call.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "brainstorming" in system_msg
        assert "Use when creating" in system_msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py::TestAgentSkillsIntegration -v`
Expected: FAIL — `AssertionError` (agent does not initialize `_skill_manager`)

- [ ] **Step 3: Integrate SkillManager into Agent**

Modify `src/air_agent/agent.py`:

**Add import** at line 12 (after existing imports):

```python
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import LLMSkillRouter
```

**Add to `__init__`** after line 28 (`self._conversations = ...`):

```python
        self._skill_manager: SkillManager | None = None
        if config.skills_dir:
            self._skill_manager = SkillManager(config.skills_dir)
            self._skill_manager.load()
            self._skill_router = LLMSkillRouter(client=self._client, model=config.model)
```

**Modify `_build_messages`** to append skill metadata to system prompt. Replace the existing `_build_messages` method (lines 75-82) with:

```python
    def _build_messages(self, user_input: str, conversation_id: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.config.system_prompt or self._skill_manager:
            system_content = self.config.system_prompt or ""
            if self._skill_manager:
                summary = self._skill_manager.metadata_summary()
                if summary:
                    system_content += f"\n\n## Available Skills\n{summary}"
            messages.append({"role": "system", "content": system_content})
        if conversation_id and conversation_id in self._conversations:
            messages.extend(self._conversations[conversation_id])
        messages.append({"role": "user", "content": user_input})
        return messages
```

**Add skill matching in `_run`** — insert after `history: list[dict[str, Any]] = list(messages)` (line 98) and before the `for` loop:

```python
        if self._skill_manager and self._skill_manager.skills:
            matched = await self._skill_router.match(
                user_input=messages[-1]["content"],
                skills=self._skill_manager.skills,
            )
            for skill in matched:
                history.insert(0, {
                    "role": "system",
                    "content": f'<skill name="{skill.name}">\n{skill.content}\n</skill>',
                })
```

**Add `self._skill_router` default** in `__init__` (for when skills_dir is not set):

```python
        self._skill_router = None
```

The full updated `__init__` should be:

```python
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.default_headers,
        )
        self._registry = ToolRegistry()
        self._mcp_clients: list[MCPClient] = []
        self._conversations: dict[str, list[dict[str, Any]]] = {}
        self._skill_manager: SkillManager | None = None
        self._skill_router = None
        if config.skills_dir:
            self._skill_manager = SkillManager(config.skills_dir)
            self._skill_manager.load()
            self._skill_router = LLMSkillRouter(client=self._client, model=config.model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/ -v`
Expected: All tests PASS (old + new)

- [ ] **Step 6: Commit**

```bash
git add src/air_agent/agent.py tests/test_skills.py
git commit -m "feat: integrate SkillManager into Agent with prompt injection"
```

---

### Task 6: Update public API exports

**Files:**
- Modify: `src/air_agent/__init__.py`

- [ ] **Step 1: Add skills exports to __init__.py**

Update `src/air_agent/__init__.py` to:

```python
"""Air Agent — lightweight AI agent library."""

from air_agent.config import AgentConfig, MCPServerStdio, MCPServerSSE, SubagentConfig
from air_agent.types import Response, StreamEvent, SubagentResult
from air_agent.agent import Agent
from air_agent.skills.skill import Skill
from air_agent.skills.manager import SkillManager
from air_agent.skills.router import SkillRouter, LLMSkillRouter

__all__ = [
    "Agent",
    "AgentConfig",
    "MCPServerStdio",
    "MCPServerSSE",
    "SubagentConfig",
    "Response",
    "StreamEvent",
    "SubagentResult",
    "Skill",
    "SkillManager",
    "SkillRouter",
    "LLMSkillRouter",
]
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/air_agent/__init__.py
git commit -m "feat: export Skills types in public API"
```

---

### Task 7: Skill matching in streaming mode

**Files:**
- Modify: `src/air_agent/agent.py` — add skill matching to `_run_stream`

- [ ] **Step 1: Write the failing test for streaming + skills**

Append to `tests/test_skills.py`:

```python
class TestStreamingWithSkills:
    def _create_skill_file(self, directory: Path, filename: str, name: str, description: str, content: str = ""):
        path = directory / filename
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n{content}\n"
        )

    @pytest.mark.asyncio
    async def test_streaming_injects_matched_skills(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill_file(
            skills_dir, "brainstorming.md", "brainstorming",
            "Use when creating",
            "Ask questions one at a time.",
        )

        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
        )
        agent = Agent(config)

        # Mock the routing call (first call) to return brainstorming
        routing_response = MagicMock()
        routing_response.choices = [MagicMock()]
        routing_response.choices[0].message.content = "brainstorming"

        # Mock the streaming response (second call)
        stream_event_1 = MagicMock()
        stream_event_1.choices = [MagicMock()]
        stream_event_1.choices[0].delta.content = "Hello"
        stream_event_1.choices[0].delta.tool_calls = None
        stream_event_1.usage = None

        stream_event_2 = MagicMock()
        stream_event_2.choices = [MagicMock()]
        stream_event_2.choices[0].delta.content = None
        stream_event_2.choices[0].delta.tool_calls = None
        stream_event_2.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)

        async def mock_stream():
            yield stream_event_1
            yield stream_event_2

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [routing_response, mock_stream()]
            stream = await agent.run("I want to brainstorm", stream=True)
            events = []
            async for event in stream:
                events.append(event)

        # Verify routing happened
        assert mock_create.call_count == 2
        # The streaming call should have skill content in its messages
        streaming_call = mock_create.call_args_list[1]
        messages = streaming_call.kwargs["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert any("brainstorming" in m["content"] for m in system_msgs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/test_skills.py::TestStreamingWithSkills -v`
Expected: FAIL — streaming does not inject matched skills

- [ ] **Step 3: Add skill matching to `_run_stream`**

In `src/air_agent/agent.py`, inside `_run_stream`, add the same skill matching logic after `history: list[dict[str, Any]] = list(messages)` and before the `_stream_generator` definition:

```python
        if self._skill_manager and self._skill_manager.skills:
            matched = await self._skill_router.match(
                user_input=messages[-1]["content"],
                skills=self._skill_manager.skills,
            )
            for skill in matched:
                history.insert(0, {
                    "role": "system",
                    "content": f'<skill name="{skill.name}">\n{skill.content}\n</skill>',
                })
```

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/chldu/Workspace/vibe-agent && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/air_agent/agent.py tests/test_skills.py
git commit -m "feat: add skill matching to streaming mode"
```

---

## Plan Self-Review

**1. Spec coverage:**
- SKILL.md parsing → Task 1 ✓
- Directory scanning → Task 2 ✓
- Metadata summary → Task 2 ✓
- Pluggable router → Task 3 ✓
- Config skills_dir → Task 4 ✓
- Agent integration (init + _build_messages + _run) → Task 5 ✓
- Public API exports → Task 6 ✓
- Streaming support → Task 7 ✓
- Error handling (missing dir, invalid files, LLM failure) → Tasks 2, 3 ✓

**2. Placeholder scan:** No TBD, TODO, or vague steps found.

**3. Type consistency:** All types, method names, and property names are consistent across tasks. `SkillManager.skills` is `list[Skill]` everywhere. `match()` returns `list[Skill]` consistently. `skills_dir: str | None` matches across config and usage.
