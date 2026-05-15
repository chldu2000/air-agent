from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from air_agent.skills.manager import SkillManager
from air_agent.skills.router import LLMSkillRouter
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