import json
import pytest
from air_agent.config import AgentConfig, MCPServerStdio, MCPServerSSE


class TestProviderConfig:
    def test_default_provider_is_none(self):
        config = AgentConfig()

        assert config.provider is None

    def test_programmatic_provider_is_preserved_by_identity(self):
        provider = object()
        config = AgentConfig(provider=provider)

        assert config.provider is provider

    def test_positional_constructor_order_is_preserved(self):
        config = AgentConfig("m", "k", "b", "sys")

        assert config.model == "m"
        assert config.api_key == "k"
        assert config.base_url == "b"
        assert config.system_prompt == "sys"
        assert config.provider is None


class TestMemoryConfig:
    def test_memory_defaults_are_disabled(self):
        config = AgentConfig()

        assert config.memory is None
        assert config.memory_enabled is False
        assert config.memory_search_limit == 5
        assert config.memory_max_chars == 4000
        assert config.memory_summary_threshold == 12

    def test_programmatic_memory_is_preserved_by_identity(self):
        memory = object()
        config = AgentConfig(memory=memory, memory_enabled=True)

        assert config.memory is memory
        assert config.memory_enabled is True

    def test_memory_scalar_options_from_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "memory_enabled": True,
            "memory_search_limit": 7,
            "memory_max_chars": 2000,
            "memory_summary_threshold": 9,
        }))

        config = AgentConfig.from_json(str(config_file))

        assert config.memory_enabled is True
        assert config.memory_search_limit == 7
        assert config.memory_max_chars == 2000
        assert config.memory_summary_threshold == 9

    def test_memory_object_from_json_is_rejected(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "memory": {"type": "file"},
        }))

        with pytest.raises(ValueError, match="memory must be configured programmatically"):
            AgentConfig.from_json(str(config_file))

    def test_memory_env_vars(self, monkeypatch):
        monkeypatch.setenv("AIR_MEMORY_ENABLED", "true")
        monkeypatch.setenv("AIR_MEMORY_SEARCH_LIMIT", "8")
        monkeypatch.setenv("AIR_MEMORY_MAX_CHARS", "2500")
        monkeypatch.setenv("AIR_MEMORY_SUMMARY_THRESHOLD", "10")

        config = AgentConfig.from_env()

        assert config.memory_enabled is True
        assert config.memory_search_limit == 8
        assert config.memory_max_chars == 2500
        assert config.memory_summary_threshold == 10


class TestFromJson:
    def test_basic_fields(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model": "gpt-4.1",
            "api_key": "sk-test",
            "system_prompt": "You are helpful.",
            "max_iterations": 30,
            "tool_timeout": 45.0,
        }))
        config = AgentConfig.from_json(str(config_file))

        assert config.model == "gpt-4.1"
        assert config.api_key == "sk-test"
        assert config.system_prompt == "You are helpful."
        assert config.max_iterations == 30
        assert config.tool_timeout == 45.0

    def test_mcp_stdio(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "mcp_servers": [
                {"command": "npx", "args": ["-y", "some-server"], "env": {"FOO": "bar"}},
            ],
        }))
        config = AgentConfig.from_json(str(config_file))

        assert len(config.mcp_servers) == 1
        server = config.mcp_servers[0]
        assert isinstance(server, MCPServerStdio)
        assert server.command == "npx"
        assert server.args == ["-y", "some-server"]
        assert server.env == {"FOO": "bar"}

    def test_mcp_sse(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "mcp_servers": [
                {"url": "http://localhost:8080/sse", "headers": {"Auth": "token"}},
            ],
        }))
        config = AgentConfig.from_json(str(config_file))

        assert len(config.mcp_servers) == 1
        server = config.mcp_servers[0]
        assert isinstance(server, MCPServerSSE)
        assert server.url == "http://localhost:8080/sse"
        assert server.headers == {"Auth": "token"}

    def test_mcp_mixed(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "mcp_servers": [
                {"command": "npx", "args": ["server-a"]},
                {"url": "http://localhost:9000/sse"},
            ],
        }))
        config = AgentConfig.from_json(str(config_file))

        assert len(config.mcp_servers) == 2
        assert isinstance(config.mcp_servers[0], MCPServerStdio)
        assert isinstance(config.mcp_servers[1], MCPServerSSE)

    def test_unknown_keys_ignored(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model": "gpt-4o",
            "unknown_field": "should be ignored",
            "another_extra": 42,
        }))
        config = AgentConfig.from_json(str(config_file))
        assert config.model == "gpt-4o"

    def test_missing_api_key_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"model": "gpt-4o"}))
        config = AgentConfig.from_json(str(config_file))

        assert config.api_key == "sk-from-env"

    def test_empty_file_uses_defaults(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        config = AgentConfig.from_json(str(config_file))

        assert config.model == "gpt-4o"
        assert config.max_iterations == 20
        assert config.mcp_servers == []

    def test_provider_string_is_accepted(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "provider": "openai",
        }))
        config = AgentConfig.from_json(str(config_file))

        assert config.provider == "openai"

    def test_non_string_provider_value_is_rejected(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "provider": {"name": "openai"},
        }))

        with pytest.raises(ValueError, match="provider must be a string or null"):
            AgentConfig.from_json(str(config_file))


class TestFromEnv:
    def test_string_fields(self, monkeypatch):
        monkeypatch.setenv("AIR_MODEL", "gpt-4.1")
        monkeypatch.setenv("AIR_API_KEY", "sk-test")
        monkeypatch.setenv("AIR_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("AIR_SYSTEM_PROMPT", "Be concise.")
        config = AgentConfig.from_env()

        assert config.model == "gpt-4.1"
        assert config.api_key == "sk-test"
        assert config.base_url == "https://api.example.com"
        assert config.system_prompt == "Be concise."

    def test_numeric_coercion(self, monkeypatch):
        monkeypatch.setenv("AIR_MAX_ITERATIONS", "30")
        monkeypatch.setenv("AIR_TOOL_TIMEOUT", "45.5")
        config = AgentConfig.from_env()

        assert config.max_iterations == 30
        assert isinstance(config.max_iterations, int)
        assert config.tool_timeout == 45.5
        assert isinstance(config.tool_timeout, float)

    def test_mcp_servers_json(self, monkeypatch):
        monkeypatch.setenv("AIR_MCP_SERVERS", json.dumps([
            {"command": "npx", "args": ["server"]},
            {"url": "http://localhost:8080/sse"},
        ]))
        config = AgentConfig.from_env()

        assert len(config.mcp_servers) == 2
        assert isinstance(config.mcp_servers[0], MCPServerStdio)
        assert isinstance(config.mcp_servers[1], MCPServerSSE)

    def test_default_headers_json(self, monkeypatch):
        monkeypatch.setenv("AIR_DEFAULT_HEADERS", json.dumps({"X-Custom": "value"}))
        config = AgentConfig.from_env()

        assert config.default_headers == {"X-Custom": "value"}

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MYAPP_MODEL", "gpt-4.1-mini")
        monkeypatch.setenv("MYAPP_API_KEY", "sk-myapp")
        config = AgentConfig.from_env(prefix="MYAPP_")

        assert config.model == "gpt-4.1-mini"
        assert config.api_key == "sk-myapp"

    def test_no_vars_returns_defaults(self, monkeypatch):
        for key in list(monkeypatch._env if hasattr(monkeypatch, '_env') else {}):
            if key.startswith("AIR_"):
                monkeypatch.delenv(key, raising=False)
        config = AgentConfig.from_env()

        assert config.model == "gpt-4o"
        assert config.max_iterations == 20
        assert config.mcp_servers == []

    def test_api_key_fallback_to_openai_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
        config = AgentConfig.from_env()

        assert config.api_key == "sk-openai-fallback"

    def test_provider_string_from_env(self, monkeypatch):
        monkeypatch.setenv("AIR_PROVIDER", "openai")
        config = AgentConfig.from_env()

        assert config.provider == "openai"


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


class TestTracingConfig:
    def test_tracing_defaults_are_disabled(self):
        config = AgentConfig()

        assert config.enable_tracing is False
        assert config.log_events is False
        assert config.event_handlers is None
        assert config.max_tool_retries == 0

    def test_tracing_config_from_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "enable_tracing": True,
            "log_events": True,
            "max_tool_retries": 2,
        }))

        config = AgentConfig.from_json(str(config_file))

        assert config.enable_tracing is True
        assert config.log_events is True
        assert config.max_tool_retries == 2

    def test_event_handlers_from_json_are_ignored(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "event_handlers": ["handler_name"],
        }))

        config = AgentConfig.from_json(str(config_file))

        assert config.event_handlers is None

    def test_tracing_config_from_env(self, monkeypatch):
        monkeypatch.setenv("AIR_ENABLE_TRACING", "true")
        monkeypatch.setenv("AIR_LOG_EVENTS", "1")
        monkeypatch.setenv("AIR_MAX_TOOL_RETRIES", "3")

        config = AgentConfig.from_env()

        assert config.enable_tracing is True
        assert config.log_events is True
        assert config.max_tool_retries == 3

    def test_tracing_env_false_values(self, monkeypatch):
        monkeypatch.setenv("AIR_ENABLE_TRACING", "false")
        monkeypatch.setenv("AIR_LOG_EVENTS", "0")

        config = AgentConfig.from_env()

        assert config.enable_tracing is False
        assert config.log_events is False

    def test_invalid_tracing_env_bool_raises(self, monkeypatch):
        monkeypatch.setenv("AIR_ENABLE_TRACING", "ture")

        with pytest.raises(ValueError, match="Invalid boolean value"):
            AgentConfig.from_env()
