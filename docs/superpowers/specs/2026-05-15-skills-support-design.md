# Skills Support Design

## Overview

Add a Skills system to air-agent that allows loading SKILL.md files from a specified directory and injecting relevant skill instructions into the agent's context via prompt injection.

## Requirements

1. Initialize agent with a `skills_dir` pointing to a directory of SKILL.md files
2. Parse SKILL.md files (YAML frontmatter + Markdown body)
3. Progressive loading: metadata always injected, full content loaded on-demand
4. Pluggable routing strategy for skill matching (default: LLM-based)

## File Structure

```
src/air_agent/
├── skills/
│   ├── __init__.py        # Exports: SkillManager, Skill, SkillRouter
│   ├── skill.py           # Skill dataclass + SKILL.md parser
│   ├── manager.py         # SkillManager: scanning, loading, routing
│   └── router.py          # SkillRouter ABC + LLMSkillRouter default
```

## Data Model

### Skill (`skill.py`)

```python
@dataclass
class Skill:
    name: str           # From YAML frontmatter
    description: str    # From YAML frontmatter
    content: str        # Markdown body (instructions)
    path: Path          # Source file path
```

### SKILL.md File Format

Each skill is a Markdown file with YAML frontmatter:

```markdown
---
name: skill-name
description: Use when [triggering conditions]
---

# Skill Name

[Instructions in Markdown]
```

**Frontmatter fields:**
- `name` (required): lowercase, hyphens, max 64 chars
- `description` (required): triggering conditions, max 1024 chars

**Discovery rules:**
- Scan `skills_dir` for all `*.md` files
- Each file is one skill
- Validate frontmatter on load, skip invalid files with warning

## SkillManager (`manager.py`)

```python
class SkillManager:
    def __init__(self, skills_dir: str | Path, router: SkillRouter | None = None):
        self.skills_dir = Path(skills_dir)
        self.skills: list[Skill] = []
        self.router = router

    def load(self) -> None:
        """Scan directory and parse all SKILL.md files"""

    def metadata_summary(self) -> str:
        """Generate compact summary of all skills for system prompt injection"""

    async def match(self, user_input: str) -> list[Skill]:
        """Route user input to relevant skills via configured strategy"""

    def get_skill(self, name: str) -> Skill | None:
        """Get skill by name"""
```

## SkillRouter (`router.py`)

```python
class SkillRouter(ABC):
    @abstractmethod
    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        """Match user input against available skills"""

class LLMSkillRouter(SkillRouter):
    """Default: use a lightweight LLM call to select relevant skills"""

    def __init__(self, client: AsyncOpenAI, model: str):
        ...

    async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
        # Send skill metadata + user_input to LLM
        # Return matched skills (or empty list if none match)
```

**LLM routing prompt:**
- System: "You are a skill router. Given a user input and a list of available skills, return the names of skills that are relevant."
- Include all skill name+description pairs
- User: the actual user input
- Response: comma-separated skill names, or "none"

## Config Extension (`config.py`)

```python
@dataclass
class AgentConfig:
    # ... existing fields ...
    skills_dir: str | None = None
```

The router is constructed internally by the Agent based on its own OpenAI client and model, so no `skills_router` field in config.

## Agent Integration (`agent.py`)

### Initialization (`__init__`)

```python
if config.skills_dir:
    self._skill_manager = SkillManager(config.skills_dir)
    self._skill_manager.load()
```

### Prompt Construction (`_build_messages`)

1. Always append skill metadata summary to system prompt:
```
## Available Skills
- brainstorming: Use when starting creative work
- debugging: Use when encountering bugs
```

2. In `_run()` before the ReAct loop, call `match()` to find relevant skills
3. Inject matched skill content as additional system message:
```
<skill name="brainstorming">
[Full Markdown content]
</skill>
```

### Run Flow (`_run`)

```
_build_messages(user_input)
  -> system_prompt + skill metadata summary + history + user_input

if skill_manager:
  matched = await skill_manager.match(user_input)
  if matched:
    inject skill content as system message

enter ReAct loop (unchanged)
```

## Error Handling

- `skills_dir` does not exist: log warning, skip skill loading
- Invalid SKILL.md (missing frontmatter): log warning, skip that file
- Router LLM call fails: return empty list (no skills injected)
- Empty skills_dir: no-op, no metadata injected

## Public API (`__init__.py`)

Add exports:
- `Skill`
- `SkillManager`
- `SkillRouter`
- `LLMSkillRouter`

## Dependencies

No new external dependencies. Uses existing `openai` client for LLM routing and stdlib `pathlib` for file scanning.

## Testing Strategy

- Unit: SKILL.md parsing with valid/invalid frontmatter
- Unit: SkillManager directory scanning
- Unit: metadata_summary generation
- Integration: LLMSkillRouter with mocked LLM response
- Integration: full Agent.run with skills_dir configured
