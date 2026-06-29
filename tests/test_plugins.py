from __future__ import annotations

import json
from pathlib import Path

import pytest

from air_agent import Agent, AgentConfig
from air_agent.memory import InMemoryMemoryStore, MemoryRecord
from air_agent.planner import Plan, PlanContext, PlanStep, StepResult
from air_agent.plugins import PluginContext, PluginLoadError, PluginManifest, load_plugin
from air_agent.providers import LLMResponse, LLMToolCall
from air_agent.tools.builtin.config import BuiltinToolsConfig
from air_agent.tools.registry import ToolRegistry


def _write_manifest(plugin_dir: Path, data: dict) -> None:
    plugin_dir.mkdir()
    (plugin_dir / "air-agent-plugin.json").write_text(json.dumps(data))


def _basic_manifest(**overrides):
    data = {
        "name": "example",
        "version": "0.1.0",
        "description": "Example plugin",
        "entrypoint": "plugin:register",
        "capabilities": ["tools"],
    }
    data.update(overrides)
    return data


class FakeCompletionProvider:
    supports_tools = True
    supports_streaming = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
                "tools": list(kwargs["tools"]) if kwargs.get("tools") else None,
            }
        )
        return self.responses.pop(0)


class NoToolProvider(FakeCompletionProvider):
    supports_tools = False


class FakePlanner:
    async def create_plan(self, goal: str, context: PlanContext) -> Plan:
        return Plan(goal=goal, steps=[PlanStep(id="step_1", description="Do it")])

    async def execute_step(self, step: PlanStep, context: PlanContext) -> StepResult:
        return StepResult(step_id=step.id, status="success", content="done")

    async def revise_plan(self, plan: Plan, result: StepResult) -> Plan:
        return plan


def test_plugin_manifest_parses_valid_manifest(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(permissions={"network": ["example.com"]}))

    manifest = PluginManifest.from_dir(plugin_dir)

    assert manifest.name == "example"
    assert manifest.version == "0.1.0"
    assert manifest.description == "Example plugin"
    assert manifest.entrypoint == "plugin:register"
    assert manifest.capabilities == ["tools"]
    assert manifest.permissions == {"network": ["example.com"]}
    assert manifest.path == plugin_dir


def test_plugin_manifest_missing_file_raises(tmp_path: Path):
    with pytest.raises(PluginLoadError, match="air-agent-plugin.json"):
        PluginManifest.from_dir(tmp_path / "missing")


@pytest.mark.parametrize("field", ["name", "version", "description", "entrypoint"])
def test_plugin_manifest_missing_required_field_raises(tmp_path: Path, field: str):
    plugin_dir = tmp_path / "plugin"
    data = _basic_manifest()
    del data[field]
    _write_manifest(plugin_dir, data)

    with pytest.raises(PluginLoadError, match=field):
        PluginManifest.from_dir(plugin_dir)


def test_plugin_manifest_invalid_capability_raises(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["tools", "telepathy"]))

    with pytest.raises(PluginLoadError, match="telepathy"):
        PluginManifest.from_dir(plugin_dir)


def test_plugin_permissions_require_explicit_authorization(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(permissions={"network": ["example.com"]}))
    (plugin_dir / "plugin.py").write_text("def register(context):\n    pass\n")

    registry = ToolRegistry()
    with pytest.raises(PluginLoadError, match="network"):
        load_plugin(plugin_dir, registry=registry, plugin_permissions=None)


def test_plugin_permissions_load_when_authorized(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(permissions={"network": ["example.com"]}))
    (plugin_dir / "plugin.py").write_text("def register(context):\n    context.metadata['loaded'] = True\n")

    registry = ToolRegistry()
    result = load_plugin(
        plugin_dir,
        registry=registry,
        plugin_permissions={"example": True},
    )

    assert result.manifest.name == "example"
    assert result.context.metadata == {"loaded": True}


def test_load_plugin_rejects_malformed_entrypoint(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(entrypoint="not-a-module-path"))

    with pytest.raises(PluginLoadError, match="entrypoint"):
        load_plugin(plugin_dir, registry=ToolRegistry(), plugin_permissions=None)


def test_load_plugin_rejects_missing_function(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(entrypoint="plugin:register"))
    (plugin_dir / "plugin.py").write_text("def other(context):\n    pass\n")

    with pytest.raises(PluginLoadError, match="register"):
        load_plugin(plugin_dir, registry=ToolRegistry(), plugin_permissions=None)


def test_load_plugin_reports_import_failure(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(entrypoint="plugin:register"))
    (plugin_dir / "plugin.py").write_text("raise RuntimeError('boom')\n")

    with pytest.raises(PluginLoadError, match="boom"):
        load_plugin(plugin_dir, registry=ToolRegistry(), plugin_permissions=None)


@pytest.mark.asyncio
async def test_plugin_context_registers_namespaced_tool(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest())
    (plugin_dir / "plugin.py").write_text(
        "async def search(query: str) -> str:\n"
        "    return 'found ' + query\n\n"
        "def register(context):\n"
        "    context.register_tool(search, namespace='web', description='Search web')\n"
    )
    registry = ToolRegistry()

    load_plugin(plugin_dir, registry=registry, plugin_permissions=None)

    assert registry.has_tool("web.search")
    result = await registry.execute("web.search", '{"query": "air"}')
    assert result == "found air"


def test_plugin_context_rejects_tool_conflict(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest())
    (plugin_dir / "plugin.py").write_text(
        "async def read_file(path: str) -> str:\n"
        "    return path\n\n"
        "def register(context):\n"
        "    context.register_tool(read_file)\n"
    )
    registry = ToolRegistry()

    async def read_file(path: str) -> str:
        return path

    registry.register(read_file, name="read_file")
    with pytest.raises(PluginLoadError, match="read_file"):
        load_plugin(plugin_dir, registry=registry, plugin_permissions=None)


def test_plugin_context_collects_extension_objects(tmp_path: Path):
    registry = ToolRegistry()
    manifest = PluginManifest(
        name="example",
        version="0.1.0",
        description="Example plugin",
        entrypoint="plugin:register",
        capabilities=[],
        permissions={},
        metadata={},
        path=tmp_path,
    )
    context = PluginContext(manifest=manifest, registry=registry)
    provider = object()
    memory = object()
    planner = object()

    context.add_skills_dir("skills")
    context.set_provider(provider)
    context.set_memory(memory)
    context.set_planner(planner)

    assert context.skills_dirs == ["skills"]
    assert context.provider is provider
    assert context.memory is memory
    assert context.planner is planner


@pytest.mark.asyncio
async def test_agent_loads_plugin_tool_and_can_call_it(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest())
    (plugin_dir / "plugin.py").write_text(
        "async def search(query: str) -> str:\n"
        "    return 'found ' + query\n\n"
        "def register(context):\n"
        "    context.register_tool(search, namespace='web', description='Search web')\n"
    )
    provider = FakeCompletionProvider([
        LLMResponse(content="", tool_calls=[LLMToolCall(id="tc_1", name="web.search", arguments='{"query": "air"}')]),
        LLMResponse(content="done"),
    ])
    agent = Agent(AgentConfig(model="fake-model", provider=provider, plugins=[str(plugin_dir)]))

    result = await agent.run("search")

    assert result.content == "done"
    assert agent._registry.has_tool("web.search")
    assert provider.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": "found air",
    }


def test_agent_plugin_tool_conflict_raises(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest())
    (plugin_dir / "plugin.py").write_text(
        "async def read_file(path: str) -> str:\n"
        "    return path\n\n"
        "def register(context):\n"
        "    context.register_tool(read_file)\n"
    )

    with pytest.raises(PluginLoadError, match="read_file"):
        Agent(AgentConfig(model="fake-model", provider=FakeCompletionProvider([]), plugins=[str(plugin_dir)]))


@pytest.mark.asyncio
async def test_plugin_tool_does_not_bypass_provider_tool_support_check(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest())
    (plugin_dir / "plugin.py").write_text(
        "async def search(query: str) -> str:\n"
        "    return query\n\n"
        "def register(context):\n"
        "    context.register_tool(search, namespace='web')\n"
    )
    agent = Agent(AgentConfig(
        model="fake-model",
        provider=NoToolProvider([]),
        builtin_tools=BuiltinToolsConfig(enabled=False),
        plugins=[str(plugin_dir)],
    ))

    with pytest.raises(RuntimeError, match="does not support tool calling"):
        await agent.run("search")


def test_agent_loads_plugin_skills_together_with_config_skills(tmp_path: Path):
    config_skills = tmp_path / "config-skills"
    plugin_skills = tmp_path / "plugin-skills"
    for root, name in [(config_skills, "config_skill"), (plugin_skills, "plugin_skill")]:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Use {name}\n---\n# {name}\n"
        )

    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["skills"]))
    (plugin_dir / "plugin.py").write_text(
        f"def register(context):\n"
        f"    context.add_skills_dir({str(plugin_skills)!r})\n"
    )
    agent = Agent(AgentConfig(
        model="fake-model",
        provider=FakeCompletionProvider([]),
        skills_dir=str(config_skills),
        plugins=[str(plugin_dir)],
    ))

    assert agent._skill_manager is not None
    names = {skill.name for skill in agent._skill_manager.skills}
    assert names == {"config_skill", "plugin_skill"}


def test_agent_uses_plugin_provider_when_no_provider_is_configured(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["provider"]))
    (plugin_dir / "plugin.py").write_text(
        "from air_agent.providers import LLMResponse\n\n"
        "class PluginProvider:\n"
        "    supports_tools = True\n"
        "    supports_streaming = False\n"
        "    async def complete(self, **kwargs):\n"
        "        return LLMResponse(content='from plugin provider')\n\n"
        "def register(context):\n"
        "    context.set_provider(PluginProvider())\n"
    )
    agent = Agent(AgentConfig(model="fake-model", plugins=[str(plugin_dir)]))

    assert agent._provider.__class__.__name__ == "PluginProvider"


def test_agent_preserves_plugin_memory_and_planner_by_identity(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["memory", "planner"]))
    (plugin_dir / "plugin.py").write_text(
        "from air_agent.memory import InMemoryMemoryStore\n\n"
        "class PluginPlanner:\n"
        "    async def create_plan(self, goal, context):\n"
        "        raise NotImplementedError\n"
        "    async def execute_step(self, step, context):\n"
        "        raise NotImplementedError\n"
        "    async def revise_plan(self, plan, result):\n"
        "        return plan\n\n"
        "memory = InMemoryMemoryStore()\n"
        "planner = PluginPlanner()\n\n"
        "def register(context):\n"
        "    context.set_memory(memory)\n"
        "    context.set_planner(planner)\n"
    )
    agent = Agent(AgentConfig(model="fake-model", provider=FakeCompletionProvider([]), plugins=[str(plugin_dir)]))

    assert isinstance(agent.config.memory, InMemoryMemoryStore)
    assert agent.config.planner.__class__.__name__ == "PluginPlanner"
    assert agent._planner is agent.config.planner


def test_agent_explicit_provider_conflicts_with_plugin_provider(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["provider"]))
    (plugin_dir / "plugin.py").write_text(
        "class PluginProvider:\n"
        "    supports_tools = True\n"
        "    supports_streaming = False\n\n"
        "def register(context):\n"
        "    context.set_provider(PluginProvider())\n"
    )

    with pytest.raises(PluginLoadError, match="provider"):
        Agent(AgentConfig(model="fake-model", provider=FakeCompletionProvider([]), plugins=[str(plugin_dir)]))


def test_agent_explicit_memory_conflicts_with_plugin_memory(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["memory"]))
    (plugin_dir / "plugin.py").write_text(
        "from air_agent.memory import InMemoryMemoryStore\n\n"
        "def register(context):\n"
        "    context.set_memory(InMemoryMemoryStore())\n"
    )

    with pytest.raises(PluginLoadError, match="memory"):
        Agent(AgentConfig(
            model="fake-model",
            provider=FakeCompletionProvider([]),
            memory=InMemoryMemoryStore([MemoryRecord(id='x', scope='global', kind='fact', content='x')]),
            plugins=[str(plugin_dir)],
        ))


def test_agent_explicit_planner_conflicts_with_plugin_planner(tmp_path: Path):
    plugin_dir = tmp_path / "plugin"
    _write_manifest(plugin_dir, _basic_manifest(capabilities=["planner"]))
    (plugin_dir / "plugin.py").write_text(
        "class PluginPlanner:\n"
        "    async def create_plan(self, goal, context):\n"
        "        raise NotImplementedError\n"
        "    async def execute_step(self, step, context):\n"
        "        raise NotImplementedError\n"
        "    async def revise_plan(self, plan, result):\n"
        "        return plan\n\n"
        "def register(context):\n"
        "    context.set_planner(PluginPlanner())\n"
    )

    with pytest.raises(PluginLoadError, match="planner"):
        Agent(AgentConfig(
            model="fake-model",
            provider=FakeCompletionProvider([]),
            planner=FakePlanner(),
            plugins=[str(plugin_dir)],
        ))
