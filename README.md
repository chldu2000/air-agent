# air-agent

[中文文档](README_zh.md)

A lightweight Python AI Agent library. Built on the OpenAI Chat Completions API with support for tool-calling loops, MCP Server connections, parallel subagents, and streaming output. Designed to be imported directly by other Python projects.

## Installation

```bash
uv add air-agent
```

Or in development mode:

```bash
git clone https://github.com/chldu2000/air-agent.git
cd air-agent
uv sync --group dev
```

## Quick Start

### Basic Conversation

```python
import asyncio
from air_agent import Agent, AgentConfig

async def main():
    agent = Agent(AgentConfig(model="gpt-4o"))
    response = await agent.run("Explain quantum computing in one sentence")
    print(response.content)

asyncio.run(main())
```

### Register Local Tools

```python
agent = Agent(AgentConfig(model="gpt-4o", api_key="sk-xxx"))

@agent.tool(name="add", description="Calculate the sum of two numbers")
async def add(a: int, b: int) -> int:
    return a + b

response = await agent.run("What is 3 plus 5?")
# The agent will automatically call the add tool and return the result
```

Parameter types are inferred from the function signature and converted to the JSON Schema required by OpenAI tool calling.

### Streaming Output

```python
async for event in await agent.run("Write a poem about programming", stream=True):
    if event.type == "text":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Calling tool: {event.name}]")
    elif event.type == "tool_result":
        print(f"[Tool result: {event.content}]")
    elif event.type == "done":
        print(f"\nDone, token usage: {event.usage}")
```

### Tracing and Structured Events

Tracing is opt-in. When enabled, the agent emits structured `RunEvent` records for LLM calls, tool calls, retries, errors, and completion.

```python
from air_agent import Agent, AgentConfig

events = []

agent = Agent(AgentConfig(
    model="gpt-4o",
    enable_tracing=True,
    log_events=True,
    event_handlers=[events.append],
))

response = await agent.run("What files are in this project?")

for event in events:
    print(event.to_dict())

tool_duration_ms = sum(
    event.duration_ms or 0
    for event in events
    if event.type == "tool_end"
)
failed_tools = [
    event
    for event in events
    if event.type == "tool_error"
]

print(f"Tool time: {tool_duration_ms:.1f}ms")
for event in failed_tools:
    print(f"Failed tool: {event.name} ({event.error_kind})")
```

Useful event types include `llm_start`, `llm_end`, `tool_start`, `tool_end`, `tool_error`, `retry`, and `done`. Tool errors include an `error_kind` such as `invalid_arguments`, `tool_not_found`, `timeout`, `permission_denied`, or `tool_error`.

Skills tracing adds:

- `skill_route_start` with `metadata.candidate_names`, `metadata.candidate_count`, and `metadata.router`
- `skill_route_end` with the router `raw output` in `content`, `metadata.matched_names`, `metadata.unrecognized_names`, and `duration_ms`
- `skill_route_error` with the failure message in `content`, `metadata.error_type`, `metadata.fallback="no_skills"`, and `duration_ms`
- `skill_injected` with the injected skill `name`, `metadata.path`, and `metadata.content_length`

`skill_route_end.content` contains the complete model-generated router output. Tracing logs may therefore include sensitive prompt or routing data; enable logging, storage, access, and retention controls accordingly.

### Multi-turn Conversation

```python
response = await agent.run("Hello", conversation_id="session-1")
response = await agent.run("What did I just say?", conversation_id="session-1")
# The second turn includes context from the first turn
```

### Load Configuration from JSON

```json
{
  "model": "gpt-4o",
  "system_prompt": "You are a coding assistant",
  "mcp_servers": [
    {"command": "npx", "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"]},
    {"url": "http://localhost:8080/sse"}
  ]
}
```

```python
config = AgentConfig.from_json("agent-config.json")
agent = Agent(config)
```

The `mcp_servers` field auto-detects the transport type based on `command` (stdio) or `url` (SSE).

### Load Configuration from Environment Variables

```bash
export AIR_MODEL=gpt-4o
export AIR_SYSTEM_PROMPT="You are an assistant"
export AIR_MAX_ITERATIONS=30
export AIR_MCP_SERVERS='[{"command":"npx","args":["server"]}]'
```

```python
config = AgentConfig.from_env()          # default AIR_ prefix
config = AgentConfig.from_env(prefix="MYAPP_")  # custom prefix
agent = Agent(config)
```

Supported environment variables:

| Variable | Type | Description |
| ---- | ---- | ---- |
| `AIR_MODEL` | str | Model name |
| `AIR_API_KEY` | str | API key (takes precedence over `OPENAI_API_KEY`) |
| `AIR_BASE_URL` | str | Custom API endpoint |
| `AIR_PROVIDER` | str | Provider name (`openai` by default) |
| `AIR_SYSTEM_PROMPT` | str | System prompt |
| `AIR_MAX_ITERATIONS` | int | Max tool-calling rounds |
| `AIR_TOOL_TIMEOUT` | float | Tool call timeout in seconds |
| `AIR_MCP_SERVERS` | JSON | MCP server list |
| `AIR_DEFAULT_HEADERS` | JSON | Custom request headers |
| `AIR_SKILLS_DIR` | str | Skills directory path |

### Custom LLM Providers

OpenAI remains the default provider. For OpenAI-compatible APIs, keep using `model`, `api_key`, `base_url`, and `default_headers`:

```python
from air_agent import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="gpt-4o",
    api_key="sk-xxx",
    base_url="https://api.example.com/v1",
    default_headers={"X-API-Key": "custom-header"},
))
```

For other backends, pass an object that implements `LLMProvider`. Provider methods return the neutral `LLMResponse` and `LLMStreamChunk` types, so you can adapt any backend without OpenAI-specific payloads.

```python
from typing import Any, AsyncIterator

from air_agent import Agent, AgentConfig, BuiltinToolsConfig, LLMResponse, LLMStreamChunk


class EchoProvider:
    supports_tools = False
    supports_streaming = True

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> LLMResponse:
        last_message = messages[-1]["content"]
        return LLMResponse(content=f"echo: {last_message}")

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(content_delta="echo: ")
        yield LLMStreamChunk(content_delta=str(messages[-1]["content"]))


agent = Agent(AgentConfig(
    model="echo",
    provider=EchoProvider(),
    builtin_tools=BuiltinToolsConfig(enabled=False),
))
```

If `supports_tools = False`, runs with registered or enabled tools fail clearly instead of silently ignoring them. Built-in tools are enabled by default, so disable them as shown above or implement tool support in your provider.

### Skills

Load skill instructions from a directory of skill folders. Each skill is a directory (kebab-case named) containing a `SKILL.md` file with YAML frontmatter for metadata and Markdown body for instructions.

**Directory structure:**

```text
skills/
├── brainstorming/
│   └── SKILL.md              # Required: metadata + instructions
├── data-analysis/
│   ├── SKILL.md
│   ├── scripts/              # Optional: executable scripts
│   │   └── process_data.py
│   └── references/           # Optional: templates, schemas
│       └── data_schema.json
```

**SKILL.md format** (`skills/brainstorming/SKILL.md`):

```markdown
---
name: brainstorming
description: Use when starting creative work or exploring ideas
---

# Brainstorming

Ask questions one at a time to refine the idea.
```

**Usage:**

```python
from air_agent import Agent, AgentConfig

config = AgentConfig(
    model="gpt-4o",
    skills_dir="./skills",  # directory containing skill subdirectories
)
agent = Agent(config)
response = await agent.run("I want to brainstorm a new feature")
```

Skills work via progressive prompt injection:
- All skill metadata (name + description) is always included in the system prompt
- When a user query matches relevant skills, the full skill content is injected into the conversation context
- Skill matching uses an LLM-based router by default; you can provide a custom `SkillRouter` implementation

**Custom router:**

```python
from air_agent import SkillRouter

class KeywordRouter(SkillRouter):
    async def match(self, user_input: str, skills: list) -> list:
        return [s for s in skills if s.name in user_input.lower()]
```

### Built-in Tools

Agent comes with a minimal built-in toolset for file system operations and shell commands. These are enabled by default and registered automatically.

| Tool | Description |
| ---- | ----------- |
| `read_file` | Read file contents with offset/limit support |
| `write_file` | Write content to a file, auto-create directories |
| `list_directory` | List directory entries with type and size info |
| `find_files` | Find files matching a glob pattern |
| `grep` | Search file contents with regex |
| `run_shell` | Execute shell commands |

**Default usage (no configuration needed):**

```python
from air_agent import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-4o", api_key="sk-xxx"))
# read_file, write_file, list_directory, find_files, grep, run_shell are all available
```

**Configuration:**

```python
from air_agent import BuiltinToolsConfig

# Disable built-in tools entirely
config = AgentConfig(model="gpt-4o", builtin_tools=BuiltinToolsConfig(enabled=False))

# Select specific tools only
config = AgentConfig(model="gpt-4o",
    builtin_tools=BuiltinToolsConfig(tools=["read_file", "grep"]))

# Custom sandbox and limits
config = AgentConfig(model="gpt-4o",
    builtin_tools=BuiltinToolsConfig(
        allowed_directories=["/project"],
        max_read_size=500_000,
        max_grep_results=50,
        default_timeout=60.0,
    ))
```

**Security features:**

- **Path sandbox** — file tools only access paths within `allowed_directories` (defaults to cwd)
- **Command blocklist** — dangerous commands (`rm -rf /`, `sudo`, `mkfs`, etc.) are blocked
- **Result limits** — configurable caps on find/grep/list results and shell output
- **Truncation notices** — when results are truncated, the agent is informed so it can refine queries

**`BuiltinToolsConfig` fields:**

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `enabled` | bool | `True` | Master switch |
| `tools` | list | `None` | Tool selection (`None` = all) |
| `allowed_directories` | list | `[]` | Sandbox dirs (empty = cwd) |
| `max_read_size` | int | `1000000` | Max file read size in bytes |
| `default_timeout` | float | `30.0` | Shell command timeout |
| `blocked_commands` | list | [...] | Blocked command patterns |
| `max_find_results` | int | `200` | Find results cap |
| `max_grep_results` | int | `100` | Grep matches cap |
| `max_list_entries` | int | `500` | Directory listing cap |
| `max_output_bytes` | int | `50000` | Shell output truncation |

### Connect to MCP Servers

```python
from air_agent import MCPServerStdio, MCPServerSSE

agent = Agent(AgentConfig(
    model="gpt-4o",
    mcp_servers=[
        MCPServerStdio(command="npx", args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]),
        MCPServerSSE(url="http://localhost:8080/mcp"),
    ],
))

async with agent:  # auto connect/disconnect MCP servers
    response = await agent.run("List files under /tmp")
```

Supports both stdio and StreamableHTTP MCP transports. Once connected, tools exposed by the server are automatically registered in the agent's tool list.

### Parallel Subagents

```python
from air_agent import SubagentConfig

results = await agent.delegate(
    tasks=[
        "Analyze the code structure in src/",
        "Check test coverage in tests/",
        "Generate a CHANGELOG",
    ],
    config=SubagentConfig(max_parallel=3, timeout=60),
)

for r in results:
    print(f"[{r.status}] {r.content[:100]}")
```

Each task runs in an independent Agent instance without interference.

## Configuration

```python
AgentConfig(
    model="gpt-4o",              # Model name
    api_key="sk-xxx",            # Or set OPENAI_API_KEY env variable
    base_url=None,               # Custom API endpoint
    system_prompt="You are an assistant",  # System prompt
    max_iterations=20,           # Max tool-calling rounds
    tool_timeout=30.0,           # Single tool call timeout (seconds)
    mcp_servers=[],              # MCP server list
    skills_dir=None,             # Skills directory path
    builtin_tools=None,          # BuiltinToolsConfig or None for defaults
)
```

## Project Structure

```text
src/air_agent/
├── __init__.py          # Public API exports
├── agent.py             # Core Agent (ReAct loop + streaming)
├── config.py            # Configuration dataclass
├── types.py             # Response, StreamEvent, SubagentResult
├── tools/
│   ├── base.py          # Tool dataclass
│   ├── registry.py      # Tool registry
│   └── builtin/
│       ├── config.py    # BuiltinToolsConfig
│       ├── _permissions.py  # Path sandbox + command blocklist
│       ├── file_tools.py    # read, write, list, find, grep
│       └── shell_tools.py   # run_shell
├── mcp/
│   ├── client.py        # MCP client (stdio + streamable_http)
│   └── tool_adapter.py  # MCP tool → OpenAI format adapter
├── skills/
│   ├── skill.py         # Skill dataclass + SKILL.md parser
│   ├── manager.py       # SkillManager (directory scanning)
│   └── router.py        # SkillRouter ABC + LLMSkillRouter
└── subagent.py          # Parallel subagent manager
```

## Dependencies

- `openai` — LLM calls and tool calling
- `mcp` — MCP protocol client
- `pydantic` — Data validation

## Development

```bash
uv sync --group dev
uv run pytest tests/ -v
```

## License

MIT
