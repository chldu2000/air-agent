from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from air_agent import SkillRouteResult
from air_agent.providers import LLMResponse, LLMStreamChunk, LLMStreamToolCallDelta, LLMToolCall
from air_agent.skills.manager import SkillManager
from air_agent.skills import SkillRouteResult as SkillsSkillRouteResult
from air_agent.skills.router import LLMSkillRouter, SkillRouter
from air_agent.skills.skill import Skill, parse_skill_file


class TestParseSkillFile:
    def test_valid_skill_file(self, tmp_path: Path):
        skill_dir = tmp_path / "brainstorming"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
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
        assert skill.skill_dir == skill_dir

    def test_missing_frontmatter_returns_none(self, tmp_path: Path):
        skill_dir = tmp_path / "no-frontmatter"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Just a heading\n\nSome text.\n")
        skill = parse_skill_file(skill_file)
        assert skill is None

    def test_missing_name_returns_none(self, tmp_path: Path):
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "description: Use when something\n"
            "---\n"
            "Content here.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is None

    def test_missing_description_returns_none(self, tmp_path: Path):
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: some-skill\n"
            "---\n"
            "Content here.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill is None

    def test_empty_content(self, tmp_path: Path):
        skill_dir = tmp_path / "empty"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
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
        skill_dir = tmp_path / "multi"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
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


class TestSkillManager:
    def _create_skill(self, skills_dir: Path, skill_name: str, description: str, content: str = ""):
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: {description}\n---\n{content}\n"
        )
        return skill_dir
        return skill_dir

    def test_load_scans_subdirectories(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "brainstorming", "Use when creating")
        self._create_skill(skills_dir, "debugging", "Use when bugs")

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 2
        names = {s.name for s in manager.skills}
        assert "brainstorming" in names
        assert "debugging" in names

    def test_load_skips_directories_without_skill_md(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "valid", "Use when valid")
        (skills_dir / "no-skill-md").mkdir()

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 1
        assert manager.skills[0].name == "valid"

    def test_load_skips_invalid_skill_md(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "valid", "Use when valid")
        invalid_dir = skills_dir / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / "SKILL.md").write_text("# No frontmatter\n")

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 1
        assert manager.skills[0].name == "valid"

    def test_load_ignores_flat_files(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "my-skill", "Use when needed")
        (skills_dir / "notes.txt").write_text("Not a skill file\n")

        manager = SkillManager(skills_dir)
        manager.load()

        assert len(manager.skills) == 1

    def test_metadata_summary(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "brainstorming", "Use when creating")
        self._create_skill(skills_dir, "debugging", "Use when bugs found")

        manager = SkillManager(skills_dir)
        manager.load()

        summary = manager.metadata_summary()
        assert "- brainstorming: Use when creating" in summary
        assert "- debugging: Use when bugs found" in summary

    def test_get_skill_by_name(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "brainstorming", "Use when creating")

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

    def test_list_attachments_excludes_skill_md_and_hidden_files(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = self._create_skill(skills_dir, "debugging", "Use when bugs", "Inspect.")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "notes.md").write_text("notes")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "helper.py").write_text("print('hi')")
        (skill_dir / ".secret").write_text("hidden")

        manager = SkillManager(skills_dir)
        manager.load()
        skill = manager.get_skill("debugging")
        assert skill is not None

        result = manager.list_attachments(skill)

        assert result["truncated"] is False
        assert result["count"] == 2
        assert result["attachments"] == [
            {"path": "references/notes.md", "type": "file", "size": 5},
            {"path": "scripts/helper.py", "type": "file", "size": 11},
        ]

    def test_list_attachments_truncates_to_limit(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = self._create_skill(skills_dir, "debugging", "Use when bugs", "Inspect.")
        for index in range(3):
            (skill_dir / f"file_{index}.txt").write_text(str(index))

        manager = SkillManager(skills_dir)
        manager.load()
        skill = manager.get_skill("debugging")
        assert skill is not None

        result = manager.list_attachments(skill, max_entries=2)

        assert result["truncated"] is True
        assert result["count"] == 3
        assert len(result["attachments"]) == 2

    def test_render_skill_payload_returns_content_and_attachments(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = self._create_skill(skills_dir, "debugging", "Use when bugs", "Inspect the failure.")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "guide.md").write_text("guide")

        manager = SkillManager(skills_dir)
        manager.load()

        payload = manager.render_skill_payload("debugging", description="Need debugging help")

        assert "# Skill: debugging" in payload
        assert "Description: Use when bugs" in payload
        assert f"Path: {skill_dir}" in payload
        assert "Requested context: Need debugging help" in payload
        assert "Inspect the failure." in payload
        assert "- references/guide.md (file, 5 bytes)" in payload

    def test_render_skill_payload_for_missing_skill_lists_available_names(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "debugging", "Use when bugs")

        manager = SkillManager(skills_dir)
        manager.load()

        payload = manager.render_skill_payload("missing")

        assert "Skill not found: missing" in payload
        assert "Available skills: debugging" in payload


class TestLLMSkillRouter:
    def _make_skill(self, name: str, description: str) -> Skill:
        fake_dir = Path("/fake") / name
        return Skill(
            name=name,
            description=description,
            content=f"# {name}\nInstructions for {name}",
            path=fake_dir / "SKILL.md",
            skill_dir=fake_dir,
        )

    class FakeRoutingProvider:
        supports_tools = True
        supports_streaming = True

        def __init__(self, content: str | Exception):
            self.content = content
            self.complete_calls = []

        async def complete(self, *, model, messages, tools=None, **options):
            self.complete_calls.append(
                {"model": model, "messages": messages, "tools": tools, "options": options}
            )
            if isinstance(self.content, Exception):
                raise self.content
            return LLMResponse(content=self.content)

        async def stream(self, *, model, messages, tools=None, **options):
            if False:
                yield

    @pytest.mark.asyncio
    async def test_route_returns_raw_output_matches_unknowns_and_dedups(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  debugging, BRAINSTORMING, unknown, debugging, UNKNOWN,  brainstorming  "
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            self._make_skill("debugging", "Use when bugs"),
            self._make_skill("deploy", "Use when deploying"),
        ]

        result = await router.route("I need to brainstorm ideas", skills)
        assert result.raw_output == "  debugging, BRAINSTORMING, unknown, debugging, UNKNOWN,  brainstorming  "
        assert [skill.name for skill in result.matched_skills] == ["brainstorming", "debugging"]
        assert result.unrecognized_names == ["unknown"]
        assert result.error_type is None
        assert result.error_message is None
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_route_preserves_legacy_positional_client_constructor(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "debugging"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(mock_client, "gpt-4o")
        skills = [self._make_skill("debugging", "Use when bugs")]

        result = await router.route("fix it", skills)

        assert [skill.name for skill in result.matched_skills] == ["debugging"]
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_treats_none_and_empty_as_successful_no_match(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("deploy", "Use when deploying")]

        result = await router.route("Tell me a joke", skills)
        assert result.raw_output == ""
        assert result.matched_skills == []
        assert result.unrecognized_names == []
        assert result.error_type is None
        assert result.error_message is None

        mock_response.choices[0].message.content = "   "
        result = await router.route("Tell me a joke", skills)
        assert result.raw_output == "   "
        assert result.matched_skills == []
        assert result.unrecognized_names == []
        assert result.error_type is None
        assert result.error_message is None

        mock_response.choices[0].message.content = "none"
        result = await router.route("Tell me a joke", skills)
        assert result.raw_output == "none"
        assert result.matched_skills == []
        assert result.unrecognized_names == []
        assert result.error_type is None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_route_returns_default_result_without_skills(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        result = await router.route("Tell me a joke", [])

        assert result == SkillRouteResult()
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_returns_error_result_on_llm_error(self, caplog: pytest.LogCaptureFixture):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("test", "Use when testing")]

        with caplog.at_level("WARNING"):
            result = await router.route("test query", skills)

        assert result.raw_output == ""
        assert result.matched_skills == []
        assert result.unrecognized_names == []
        assert result.error_type == "Exception"
        assert result.error_message == "API error"
        assert result.duration_ms is not None
        assert result.duration_ms >= 0
        assert any(record.levelname == "WARNING" for record in caplog.records)

    @pytest.mark.asyncio
    async def test_route_returns_error_result_for_malformed_content(self, caplog: pytest.LogCaptureFixture):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ["brainstorming"]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("brainstorming", "Use when creating")]

        with caplog.at_level("WARNING"):
            result = await router.route("help me brainstorm", skills)

        assert result.raw_output == ""
        assert result.matched_skills == []
        assert result.unrecognized_names == []
        assert result.error_type == "AttributeError"
        assert "strip" in (result.error_message or "")
        assert result.duration_ms is not None
        assert result.duration_ms >= 0
        assert any(record.levelname == "WARNING" for record in caplog.records)

    @pytest.mark.asyncio
    async def test_route_preserves_duplicate_candidate_skills_for_same_name(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "debugging"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        first_debugging = self._make_skill("debugging", "Use when backend bugs")
        second_debugging = self._make_skill("debugging", "Use when frontend bugs")
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            first_debugging,
            second_debugging,
        ]

        result = await router.route("fix it", skills)

        assert result.raw_output == "debugging"
        assert result.unrecognized_names == []
        assert result.matched_skills == [first_debugging, second_debugging]

    @pytest.mark.asyncio
    async def test_route_keeps_candidate_name_matching_case_sensitive(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "debugging"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        router = LLMSkillRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("Debugging", "Use when bugs")]

        result = await router.route("fix it", skills)

        assert result.raw_output == "debugging"
        assert result.matched_skills == []
        assert result.unrecognized_names == ["debugging"]

    @pytest.mark.asyncio
    async def test_route_uses_custom_match_override_without_calling_llm(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        class MatchOverrideRouter(LLMSkillRouter):
            def __init__(self):
                super().__init__(client=mock_client, model="gpt-4o")
                self.match_calls: list[tuple[str, list[Skill]]] = []

            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                self.match_calls.append((user_input, skills))
                return [skills[-1]]

        router = MatchOverrideRouter()
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            self._make_skill("debugging", "Use when bugs"),
        ]

        result = await router.route("fix it", skills)

        assert result.matched_skills == [skills[-1]]
        assert len(router.match_calls) == 1
        assert router.match_calls[0] == ("fix it", skills)
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_supports_custom_match_override_that_calls_super_match(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "debugging, unknown"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        class SuperMatchRouter(LLMSkillRouter):
            def __init__(self):
                super().__init__(client=mock_client, model="gpt-4o")
                self.match_calls = 0

            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                self.match_calls += 1
                return await super().match(user_input, skills)

        router = SuperMatchRouter()
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            self._make_skill("debugging", "Use when bugs"),
        ]

        result = await router.route("fix it", skills)

        assert [skill.name for skill in result.matched_skills] == ["debugging"]
        assert router.match_calls == 1
        assert result.raw_output == "debugging, unknown"
        assert result.unrecognized_names == ["unknown"]
        assert result.error_type is None
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_preserves_super_match_error_result(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API down")
        )

        class SuperMatchRouter(LLMSkillRouter):
            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                return await super().match(user_input, skills)

        router = SuperMatchRouter(client=mock_client, model="gpt-4o")
        skills = [self._make_skill("debugging", "Use when bugs")]

        result = await router.route("fix it", skills)

        assert result.matched_skills == []
        assert result.raw_output == ""
        assert result.unrecognized_names == []
        assert result.error_type == "RuntimeError"
        assert result.error_message == "API down"
        assert result.duration_ms is not None

    @pytest.mark.asyncio
    async def test_concurrent_super_match_overrides_each_use_custom_match(self):
        mock_client = MagicMock()
        response_one = MagicMock()
        response_one.choices = [MagicMock()]
        response_one.choices[0].message.content = "debugging"
        response_two = MagicMock()
        response_two.choices = [MagicMock()]
        response_two.choices[0].message.content = "debugging"
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[response_one, response_two]
        )

        class ConcurrentSuperMatchRouter(LLMSkillRouter):
            def __init__(self):
                super().__init__(client=mock_client, model="gpt-4o")
                self.match_calls = 0
                self.both_matches_started = asyncio.Event()

            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                self.match_calls += 1
                if self.match_calls == 2:
                    self.both_matches_started.set()
                await self.both_matches_started.wait()
                return await super().match(user_input, skills)

        router = ConcurrentSuperMatchRouter()
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            self._make_skill("debugging", "Use when bugs"),
        ]

        results = await asyncio.wait_for(
            asyncio.gather(
                router.route("fix it", skills),
                router.route("fix it too", skills),
            ),
            timeout=1,
        )

        assert router.match_calls == 2
        assert [
            [skill.name for skill in result.matched_skills]
            for result in results
        ] == [["debugging"], ["debugging"]]
        assert all(result.error_type is None for result in results)
        assert mock_client.chat.completions.create.await_count == 2

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
        names = [s.name for s in result]
        assert names == ["brainstorming", "debugging"]

    @pytest.mark.asyncio
    async def test_match_uses_provider_complete(self):
        provider = self.FakeRoutingProvider("brainstorming, debugging")
        router = LLMSkillRouter(provider=provider, model="router-model")
        skills = [
            self._make_skill("brainstorming", "Use when creating"),
            self._make_skill("debugging", "Use when bugs"),
            self._make_skill("deploy", "Use when deploying"),
        ]

        result = await router.match("I need to brainstorm ideas", skills)

        assert [skill.name for skill in result] == ["brainstorming", "debugging"]
        assert provider.complete_calls[0]["model"] == "router-model"
        assert provider.complete_calls[0]["tools"] is None
        assert provider.complete_calls[0]["options"]["max_tokens"] == 100

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

    @pytest.mark.asyncio
    async def test_match_returns_empty_on_provider_error(self):
        provider = self.FakeRoutingProvider(RuntimeError("boom"))
        router = LLMSkillRouter(provider=provider, model="router-model")
        skills = [self._make_skill("test", "Use when testing")]

        result = await router.match("test query", skills)
        assert result == []


class TestSkillRouteResultAndRouting:
    def _make_skill(self, name: str, description: str) -> Skill:
        fake_dir = Path("/fake") / name
        return Skill(
            name=name,
            description=description,
            content=f"# {name}\nInstructions for {name}",
            path=fake_dir / "SKILL.md",
            skill_dir=fake_dir,
        )

    def test_skill_route_result_is_exported_from_top_level(self):
        assert SkillRouteResult.__name__ == "SkillRouteResult"
        assert SkillRouteResult is SkillsSkillRouteResult

    @pytest.mark.asyncio
    async def test_route_wraps_legacy_match_with_duration(self):
        class LegacyRouter(SkillRouter):
            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                return [skills[0]]

        router = LegacyRouter()
        skill = self._make_skill("brainstorming", "Use when creating")

        result = await router.route("help me brainstorm", [skill])

        assert result.matched_skills == [skill]
        assert result.raw_output == ""
        assert result.error_type is None
        assert result.error_message is None
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_route_returns_error_result_when_match_raises(self):
        class FailingRouter(SkillRouter):
            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                raise RuntimeError("boom")

        router = FailingRouter()
        skill = self._make_skill("brainstorming", "Use when creating")

        result = await router.route("help me brainstorm", [skill])

        assert result.matched_skills == []
        assert result.raw_output == ""
        assert result.error_type == "RuntimeError"
        assert result.error_message == "boom"
        assert result.duration_ms is not None
        assert result.duration_ms >= 0


from air_agent.agent import Agent
from air_agent.config import AgentConfig


class TestAgentSkillsIntegration:
    def _create_skill(self, skills_dir: Path, skill_name: str, description: str, content: str = ""):
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: {description}\n---\n{content}\n"
        )
        return skill_dir

    def test_agent_initializes_skill_manager(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "test-skill", "Use when testing")

        config = AgentConfig(model="gpt-4o", api_key="test-key", skills_dir=str(skills_dir))
        agent = Agent(config)

        assert agent._skill_manager is not None
        assert len(agent._skill_manager.skills) == 1
        assert agent._registry.has_tool("use_skill")

    def test_agent_without_skills_dir_has_no_manager(self):
        config = AgentConfig(model="gpt-4o", api_key="test-key")
        agent = Agent(config)
        assert agent._skill_manager is None

    @pytest.mark.asyncio
    async def test_skill_metadata_injected_into_system_prompt(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "brainstorming", "Use when creating")
        self._create_skill(skills_dir, "debugging", "Use when bugs")

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
            mock_create.return_value = mock_response
            result = await agent.run("hello")

        assert result.content == "done"
        actual_call = mock_create.call_args
        messages = actual_call.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "brainstorming" in system_msg
        assert "Use when creating" in system_msg
        assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_skill_content_is_not_automatically_injected(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "test-skill", "Use when testing", "Do the thing.")

        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
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
            mock_create.return_value = mock_response

            await agent.run("test")

        actual_call = mock_create.call_args
        messages = actual_call.kwargs["messages"]
        assert not any("Do the thing." in m.get("content", "") for m in messages)
        assert not any("<skill name=" in m.get("content", "") for m in messages)

    @pytest.mark.asyncio
    async def test_use_skill_tool_returns_payload_and_emits_events(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        debugging_dir = self._create_skill(
            skills_dir,
            "debugging",
            "Use when bugs",
            "Inspect the failure.",
        )
        (debugging_dir / "references").mkdir()
        (debugging_dir / "references" / "guide.md").write_text("guide")
        trace_events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[trace_events.append],
        )
        agent = Agent(config)

        tool_call = MagicMock()
        tool_call.id = "tc_1"
        tool_call.function.name = "use_skill"
        tool_call.function.arguments = '{"name": "debugging", "description": "Need help"}'
        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = None
        first_response.choices[0].message.tool_calls = [tool_call]
        first_response.usage = None
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "used skill"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [first_response, final_response]
            result = await agent.run("fix it", conversation_id="conv_1")

        assert result.content == "used skill"
        assert [event.type for event in trace_events] == [
            "llm_start",
            "llm_end",
            "tool_start",
            "skill_used",
            "tool_end",
            "llm_start",
            "llm_end",
            "done",
        ]
        skill_event = trace_events[3]
        assert skill_event.conversation_id == "conv_1"
        assert skill_event.name == "debugging"
        assert skill_event.metadata == {
            "skill_name": "debugging",
            "path": str(debugging_dir),
            "content_length": len("Inspect the failure."),
            "attachment_count": 1,
            "truncated": False,
        }

        tools = mock_create.call_args_list[0].kwargs["tools"]
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "use_skill" in tool_names
        second_messages = mock_create.call_args_list[1].kwargs["messages"]
        tool_messages = [message for message in second_messages if message["role"] == "tool"]
        assert len(tool_messages) == 1
        assert "# Skill: debugging" in tool_messages[0]["content"]
        assert "Inspect the failure." in tool_messages[0]["content"]
        assert "- references/guide.md (file, 5 bytes)" in tool_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_use_skill_missing_skill_returns_tool_result_without_error_event(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "debugging", "Use when bugs")
        trace_events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[trace_events.append],
        )
        agent = Agent(config)

        tool_call = MagicMock()
        tool_call.id = "tc_1"
        tool_call.function.name = "use_skill"
        tool_call.function.arguments = '{"name": "missing"}'
        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = None
        first_response.choices[0].message.tool_calls = [tool_call]
        first_response.usage = None
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "handled"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [first_response, final_response]
            result = await agent.run("fix it")

        assert result.content == "handled"
        assert "skill_used" not in [event.type for event in trace_events]
        assert "tool_error" not in [event.type for event in trace_events]
        second_messages = mock_create.call_args_list[1].kwargs["messages"]
        tool_message = [message for message in second_messages if message["role"] == "tool"][0]
        assert "Skill not found: missing" in tool_message["content"]
        assert "Available skills: debugging" in tool_message["content"]

    @pytest.mark.asyncio
    async def test_no_skill_route_events_are_emitted_by_default(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "debugging", "Use when bugs")
        trace_events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[trace_events.append],
        )
        agent = Agent(config)

        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "done"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = final_response
            await agent.run("hello")

        assert "skill_route_start" not in [event.type for event in trace_events]
        assert "skill_route_end" not in [event.type for event in trace_events]
        assert "skill_injected" not in [event.type for event in trace_events]
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_skills_events_are_not_dispatched_when_tracing_disabled(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "debugging", "Use when bugs")
        events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=False,
            event_handlers=[events.append],
        )
        agent = Agent(config)

        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "done"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = final_response
            result = await agent.run("fix it")

        assert result.content == "done"
        assert events == []


class TestStreamingWithSkills:
    def _create_skill(self, skills_dir: Path, skill_name: str, description: str, content: str = ""):
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: {description}\n---\n{content}\n"
        )
        return skill_dir

    class FakeStreamingProvider:
        supports_tools = True
        supports_streaming = True

        def __init__(self, streams: list[list[LLMStreamChunk]]):
            self.streams = list(streams)
            self.calls = []

        async def complete(self, **kwargs):
            raise AssertionError("complete should not be called in streaming skill test")

        async def stream(self, **kwargs):
            self.calls.append(
                {
                    **kwargs,
                    "messages": [dict(message) for message in kwargs["messages"]],
                    "tools": list(kwargs["tools"]) if kwargs.get("tools") else None,
                }
            )
            chunks = self.streams.pop(0)
            for chunk in chunks:
                yield chunk

    @pytest.mark.asyncio
    async def test_streaming_can_use_skill_as_regular_tool(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = self._create_skill(
            skills_dir, "brainstorming",
            "Use when creating",
            "Ask questions one at a time.",
        )
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "guide.md").write_text("guide")

        events = []
        provider = self.FakeStreamingProvider([
            [
                LLMStreamChunk(
                    tool_call_deltas=[
                        LLMStreamToolCallDelta(
                            index=0,
                            id="tc_1",
                            name="use_skill",
                            arguments='{"name":"brainstorming"}',
                        )
                    ]
                )
            ],
            [LLMStreamChunk(content_delta="Used it.")],
        ])
        config = AgentConfig(
            model="gpt-4o",
            provider=provider,
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[events.append],
        )
        agent = Agent(config)

        stream = await agent.run("I want to brainstorm", stream=True)
        stream_events = []
        async for event in stream:
            stream_events.append(event)

        assert [event.type for event in events] == [
            "llm_start",
            "llm_end",
            "tool_start",
            "skill_used",
            "tool_end",
            "llm_start",
            "llm_end",
            "done",
        ]
        skill_event = events[3]
        assert skill_event.name == "brainstorming"
        assert skill_event.metadata == {
            "skill_name": "brainstorming",
            "path": str(skills_dir / "brainstorming"),
            "content_length": len("Ask questions one at a time."),
            "attachment_count": 1,
            "truncated": False,
        }
        assert [event.type for event in stream_events] == ["tool_call", "tool_result", "text", "done"]
        assert stream_events[1].name == "use_skill"
        assert "# Skill: brainstorming" in stream_events[1].content
        assert "- references/guide.md (file, 5 bytes)" in stream_events[1].content

        assert len(provider.calls) == 2
        assert "use_skill" in [tool["function"]["name"] for tool in provider.calls[0]["tools"]]
        assert not any("Ask questions one at a time." in m.get("content", "") for m in provider.calls[0]["messages"])
        tool_messages = [m for m in provider.calls[1]["messages"] if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert "Ask questions one at a time." in tool_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_streaming_does_not_route_skills_before_first_stream_call(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(
            skills_dir, "debugging",
            "Use when bugs",
            "Inspect the failure.",
        )

        events = []
        provider = self.FakeStreamingProvider([
            [LLMStreamChunk(content_delta="Recovered")]
        ])
        config = AgentConfig(
            model="gpt-4o",
            provider=provider,
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[events.append],
        )
        agent = Agent(config)

        stream = await agent.run("fix it", stream=True)
        stream_events = []
        async for event in stream:
            stream_events.append(event)

        assert [event.type for event in events] == [
            "llm_start",
            "llm_end",
            "done",
        ]
        assert [event.type for event in stream_events] == ["text", "done"]
        assert len(provider.calls) == 1
