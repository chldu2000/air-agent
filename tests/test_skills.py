from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from air_agent import SkillRouteResult
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
        mock_response.choices[0].message.content = "debugging"
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
        assert result.error_type is None
        mock_client.chat.completions.create.assert_awaited_once()

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

    @pytest.mark.asyncio
    async def test_skill_path_injected_into_context(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = self._create_skill(skills_dir, "test-skill", "Use when testing", "Do the thing.")

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
            routing_response = MagicMock()
            routing_response.choices = [MagicMock()]
            routing_response.choices[0].message.content = "test-skill"
            mock_create.side_effect = [routing_response, mock_response]

            await agent.run("test")

        # Matched skill should include path attribute
        actual_call = mock_create.call_args_list[1]
        messages = actual_call.kwargs["messages"]
        skill_msgs = [m for m in messages if "skill name" in m.get("content", "")]
        assert len(skill_msgs) >= 1
        assert f'path="{skill_dir}"' in skill_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_skill_routing_emits_events_and_injects_in_current_order(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        brainstorming_dir = self._create_skill(
            skills_dir,
            "brainstorming",
            "Use when creating",
            "Ask questions one at a time.",
        )
        debugging_dir = self._create_skill(
            skills_dir,
            "debugging",
            "Use when bugs",
            "Inspect the failure.",
        )
        trace_events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[trace_events.append],
        )
        agent = Agent(config)

        routing_response = MagicMock()
        routing_response.choices = [MagicMock()]
        routing_response.choices[0].message.content = "brainstorming, debugging, unknown"
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "done"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [routing_response, final_response]
            result = await agent.run("create something", conversation_id="conv_1")

        assert result.content == "done"
        assert [event.type for event in trace_events] == [
            "skill_route_start",
            "skill_route_end",
            "skill_injected",
            "skill_injected",
            "llm_start",
            "llm_end",
            "done",
        ]
        route_start, route_end = trace_events[:2]
        injected_events = trace_events[2:4]
        assert route_start.conversation_id == "conv_1"
        assert route_start.metadata == {
            "candidate_names": ["brainstorming", "debugging"],
            "candidate_count": 2,
            "router": "LLMSkillRouter",
        }
        assert route_end.content == "brainstorming, debugging, unknown"
        assert route_end.duration_ms is not None
        assert route_end.metadata == {
            "matched_names": ["brainstorming", "debugging"],
            "unrecognized_names": ["unknown"],
        }
        assert [event.name for event in injected_events] == ["brainstorming", "debugging"]
        assert injected_events[0].metadata == {
            "path": str(brainstorming_dir),
            "content_length": len("Ask questions one at a time."),
        }
        assert injected_events[1].metadata == {
            "path": str(debugging_dir),
            "content_length": len("Inspect the failure."),
        }

        actual_call = mock_create.call_args_list[1]
        messages = actual_call.kwargs["messages"]
        injected_skill_messages = [
            message["content"]
            for message in messages[:2]
            if message["role"] == "system" and "skill name" in message.get("content", "")
        ]
        assert injected_skill_messages == [
            f'<skill name="debugging" path="{debugging_dir}">\nInspect the failure.\n</skill>',
            f'<skill name="brainstorming" path="{brainstorming_dir}">\nAsk questions one at a time.\n</skill>',
        ]

    @pytest.mark.asyncio
    async def test_skill_routing_error_emits_event_and_falls_back(self, tmp_path: Path):
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
        final_response.choices[0].message.content = "fallback response"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [RuntimeError("API unavailable"), final_response]
            result = await agent.run("fix it")

        assert result.content == "fallback response"
        assert [event.type for event in trace_events] == [
            "skill_route_start",
            "skill_route_error",
            "llm_start",
            "llm_end",
            "done",
        ]
        route_error = trace_events[1]
        assert route_error.content == "API unavailable"
        assert route_error.duration_ms is not None
        assert route_error.metadata == {
            "error_type": "RuntimeError",
            "fallback": "no_skills",
        }

    @pytest.mark.asyncio
    async def test_no_skill_match_emits_route_end_without_injection(self, tmp_path: Path):
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

        routing_response = MagicMock()
        routing_response.choices = [MagicMock()]
        routing_response.choices[0].message.content = "none"
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "done"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [routing_response, final_response]
            await agent.run("hello")

        assert [event.type for event in trace_events[:2]] == [
            "skill_route_start",
            "skill_route_end",
        ]
        assert not any(event.type == "skill_injected" for event in trace_events)

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

        routing_response = MagicMock()
        routing_response.choices = [MagicMock()]
        routing_response.choices[0].message.content = "debugging"
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "done"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [routing_response, final_response]
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

    @pytest.mark.asyncio
    async def test_streaming_injects_matched_skills(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(
            skills_dir, "brainstorming",
            "Use when creating",
            "Ask questions one at a time.",
        )

        events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[events.append],
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
            stream_events = []
            async for event in stream:
                stream_events.append(event)

        assert [event.type for event in events[:3]] == [
            "skill_route_start",
            "skill_route_end",
            "skill_injected",
        ]
        route_start, route_end, injected = events[:3]
        assert route_start.metadata == {
            "candidate_names": ["brainstorming"],
            "candidate_count": 1,
            "router": "LLMSkillRouter",
        }
        assert route_end.content == "brainstorming"
        assert route_end.metadata == {
            "matched_names": ["brainstorming"],
            "unrecognized_names": [],
        }
        assert injected.name == "brainstorming"
        assert injected.metadata == {
            "path": str(skills_dir / "brainstorming"),
            "content_length": len("Ask questions one at a time."),
        }
        assert [event.type for event in stream_events] == ["text", "done"]

        # Verify routing happened
        assert mock_create.call_count == 2
        # The streaming call should have skill content in its messages
        streaming_call = mock_create.call_args_list[1]
        messages = streaming_call.kwargs["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert any("brainstorming" in m["content"] for m in system_msgs)

    @pytest.mark.asyncio
    async def test_streaming_skill_routing_error_falls_back_without_skills(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(
            skills_dir, "debugging",
            "Use when bugs",
            "Inspect the failure.",
        )

        events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[events.append],
        )
        agent = Agent(config)

        stream_event = MagicMock()
        stream_event.choices = [MagicMock()]
        stream_event.choices[0].delta.content = "Recovered"
        stream_event.choices[0].delta.tool_calls = None
        stream_event.usage = MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7)

        async def mock_stream():
            yield stream_event

        with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [RuntimeError("API unavailable"), mock_stream()]
            stream = await agent.run("fix it", stream=True)
            stream_events = []
            async for event in stream:
                stream_events.append(event)

        assert [event.type for event in events] == [
            "skill_route_start",
            "skill_route_error",
            "llm_start",
            "llm_end",
            "done",
        ]
        assert events[0].metadata == {
            "candidate_names": ["debugging"],
            "candidate_count": 1,
            "router": "LLMSkillRouter",
        }
        assert events[1].content == "API unavailable"
        assert events[1].metadata == {
            "error_type": "RuntimeError",
            "fallback": "no_skills",
        }
        assert [event.type for event in stream_events] == ["text", "done"]

    @pytest.mark.asyncio
    async def test_run_falls_back_when_custom_route_implementation_raises(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "debugging", "Use when bugs", "Inspect the failure.")

        events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[events.append],
        )
        agent = Agent(config)

        class RaisingRouteRouter(SkillRouter):
            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                return []

            async def route(self, user_input: str, skills: list[Skill]) -> SkillRouteResult:
                raise RuntimeError("route boom")

        agent._skill_router = RaisingRouteRouter()

        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "fallback response"
        final_response.choices[0].message.tool_calls = None
        final_response.usage = None

        with caplog.at_level("WARNING"):
            with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = final_response
                result = await agent.run("fix it")

        assert result.content == "fallback response"
        assert [event.type for event in events] == [
            "skill_route_start",
            "skill_route_error",
            "llm_start",
            "llm_end",
            "done",
        ]
        assert events[1].content == "route boom"
        assert events[1].duration_ms is not None
        assert events[1].duration_ms >= 0
        assert events[1].metadata == {
            "error_type": "RuntimeError",
            "fallback": "no_skills",
        }
        assert any(record.levelname == "WARNING" for record in caplog.records)
        create_messages = mock_create.call_args.kwargs["messages"]
        assert not any("skill name" in message.get("content", "") for message in create_messages)

    @pytest.mark.asyncio
    async def test_streaming_falls_back_when_custom_route_implementation_raises(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._create_skill(skills_dir, "debugging", "Use when bugs", "Inspect the failure.")

        events = []
        config = AgentConfig(
            model="gpt-4o",
            api_key="test-key",
            skills_dir=str(skills_dir),
            enable_tracing=True,
            event_handlers=[events.append],
        )
        agent = Agent(config)

        class RaisingRouteRouter(SkillRouter):
            async def match(self, user_input: str, skills: list[Skill]) -> list[Skill]:
                return []

            async def route(self, user_input: str, skills: list[Skill]) -> SkillRouteResult:
                raise RuntimeError("route boom")

        agent._skill_router = RaisingRouteRouter()

        stream_event = MagicMock()
        stream_event.choices = [MagicMock()]
        stream_event.choices[0].delta.content = "Recovered"
        stream_event.choices[0].delta.tool_calls = None
        stream_event.usage = MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7)

        async def mock_stream():
            yield stream_event

        with caplog.at_level("WARNING"):
            with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_stream()
                stream = await agent.run("fix it", stream=True)
                stream_events = []
                async for event in stream:
                    stream_events.append(event)

        assert [event.type for event in events] == [
            "skill_route_start",
            "skill_route_error",
            "llm_start",
            "llm_end",
            "done",
        ]
        assert events[1].content == "route boom"
        assert events[1].duration_ms is not None
        assert events[1].duration_ms >= 0
        assert events[1].metadata == {
            "error_type": "RuntimeError",
            "fallback": "no_skills",
        }
        assert [event.type for event in stream_events] == ["text", "done"]
        assert any(record.levelname == "WARNING" for record in caplog.records)
        create_messages = mock_create.call_args.kwargs["messages"]
        assert not any("skill name" in message.get("content", "") for message in create_messages)
