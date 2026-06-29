# Skills `use_skill` Tool Implementation Plan

## Summary

Change skills from automatic router-driven prompt injection to explicit main-agent loading. Skill metadata remains in the initial system prompt, while full `SKILL.md` content and attachment metadata are returned through a built-in `use_skill` tool as ordinary tool output.

## Key Changes

- Keep `SkillManager.metadata_summary()` in the system prompt so the model can see available skill names and descriptions.
- Stop default Agent skill routing and system-message injection across ReAct, streaming, and Plan-and-Execute runs.
- Register `use_skill(name: str, description: str | None = None)` whenever skills are loaded.
- Return skill name, description, path, full instructions, and a bounded attachment manifest from `use_skill`.
- Preserve `SkillRouter`, `LLMSkillRouter`, and `SkillRouteResult` exports for legacy or advanced integrations.
- Emit normal tool events plus `skill_used` when `use_skill` successfully loads a skill.

## Implementation Notes

- Attachment manifests recursively list non-hidden files under the skill directory, excluding `SKILL.md`.
- The manifest includes relative path, type, and size, capped at 100 entries with a truncation marker.
- Missing skills return a readable tool result with available names instead of raising an execution error.
- `use_skill` conflicts with an already-registered tool of the same name during Agent initialization.

## Verification

- Focused tests:
  - `uv run pytest tests/test_skills.py tests/test_agent.py -q`
- Full suite:
  - `uv run pytest -q`
- Diff hygiene:
  - `git diff --check`
