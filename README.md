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
| `AIR_SYSTEM_PROMPT` | str | System prompt |
| `AIR_MAX_ITERATIONS` | int | Max tool-calling rounds |
| `AIR_TOOL_TIMEOUT` | float | Tool call timeout in seconds |
| `AIR_MCP_SERVERS` | JSON | MCP server list |
| `AIR_DEFAULT_HEADERS` | JSON | Custom request headers |
| `AIR_SKILLS_DIR` | str | Skills directory path |

### Skills

Load skill instructions from a directory of Markdown files. Skills use YAML frontmatter for metadata and Markdown body for instructions.

**Skill file format** (`skills/brainstorming.md`):

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
    skills_dir="./skills",  # directory containing skill .md files
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
│   └── registry.py      # Tool registry
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
