# air-agent

[English](README.md)

轻量级 Python AI Agent 库。默认使用 OpenAI Provider，同时支持自定义 LLM Provider；支持默认 ReAct 与可选 Plan-and-Execute 策略，内置工具调用循环、文件/Shell 工具、MCP Server 连接、Skills、并行 Subagent、Tracing 和流式输出。设计为可被其他 Python 项目直接引用。

## 安装

```bash
uv add air-agent
```

或开发模式：

```bash
git clone https://github.com/chldu2000/air-agent.git
cd air-agent
uv sync --group dev
```

## 快速开始

### 1. 设置 API Key

默认 OpenAI Provider 可以读取 `OPENAI_API_KEY`，也可以在 `AgentConfig` 中传入 `api_key`。

```bash
export OPENAI_API_KEY=sk-...
```

### 2. 运行基础对话

```python
import asyncio
from air_agent import Agent, AgentConfig


async def main():
    agent = Agent(AgentConfig(model="gpt-4o"))
    response = await agent.run("用一句话解释量子计算")
    print(response.content)


asyncio.run(main())
```

### 3. 注册本地工具

内置工具会自动注册；你也可以把本地 Python 函数注册为工具。

```python
import asyncio
from air_agent import Agent, AgentConfig


async def main():
    agent = Agent(AgentConfig(model="gpt-4o"))

    @agent.tool(name="add", description="计算两个数的和")
    async def add(a: int, b: int) -> int:
        return a + b

    response = await agent.run("3 加 5 等于多少？")
    print(response.content)


asyncio.run(main())
```

参数类型从函数签名自动推导，生成 OpenAI tool calling 所需的 JSON Schema。

### 4. 用 Plan-and-Execute 处理多步骤任务

默认策略是 `react`。对于较大的任务，可以显式选择 v0.6 的非流式 Plan-and-Execute MVP：Agent 会先让当前 Provider 生成有上限的 JSON plan，再通过现有的工具调用循环逐步执行，最后汇总成面向用户的答案。

```python
import asyncio
from air_agent import Agent, AgentConfig


async def main():
    agent = Agent(AgentConfig(model="gpt-4o", max_plan_steps=6))
    response = await agent.run(
        "检查这个项目，并给出接下来三个改进建议",
        strategy="plan_execute",
    )
    print(response.content)


asyncio.run(main())
```

Plan-and-Execute 在 v0.6 中需要显式开启，并且不支持 `stream=True`。

### 5. 流式输出

```python
import asyncio
from air_agent import Agent, AgentConfig


async def main():
    agent = Agent(AgentConfig(model="gpt-4o"))

    async for event in await agent.run("写一首关于编程的短诗", stream=True):
        if event.type == "text":
            print(event.content, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[调用工具: {event.name}]")
        elif event.type == "tool_result":
            print(f"\n[工具结果: {event.content}]")
        elif event.type == "done":
            print(f"\n完成，token 用量: {event.usage}")


asyncio.run(main())
```

### 6. 保留多轮对话上下文

多轮对话传入相同的 `conversation_id` 即可。air-agent 会为该 id 保留最近的对话历史。

```python
import asyncio
from air_agent import Agent, AgentConfig


async def main():
    agent = Agent(AgentConfig(model="gpt-4o"))

    first = await agent.run("我的项目名是 air-agent。", conversation_id="session-1")
    second = await agent.run("我的项目叫什么？", conversation_id="session-1")

    print(first.content)
    print(second.content)


asyncio.run(main())
```

### 7. 使用可选 Memory

Memory 默认关闭。需要同时挂载 memory store，并设置 `memory_enabled=True` 才会启用。检索到的 memory 会作为单独的 system message 注入，并标记为上下文笔记；它不会被当作用户指令。

```python
import asyncio
from air_agent import Agent, AgentConfig, InMemoryMemoryStore, MemoryRecord


async def main():
    memory = InMemoryMemoryStore()
    memory.add(MemoryRecord(
        id="project-name",
        scope="global",
        kind="fact",
        content="用户的项目名是 air-agent。",
    ))

    agent = Agent(AgentConfig(
        model="gpt-4o",
        memory=memory,
        memory_enabled=True,
    ))

    response = await agent.run("我的项目叫什么？")
    print(response.content)


asyncio.run(main())
```

如果需要跨进程持久化，可以把 `InMemoryMemoryStore()` 换成 `FileMemoryStore("memory.json")`。Memory record 支持 `fact`、`summary` 和 `task_state` 三种 kind。

### 8. 使用 Tracing 观察运行过程

Tracing 默认关闭。启用后，Agent 会为 LLM 调用、工具调用、重试、Skill 路由、错误和完成状态输出结构化 `RunEvent` 记录。

```python
from air_agent import Agent, AgentConfig

events = []

agent = Agent(AgentConfig(
    model="gpt-4o",
    enable_tracing=True,
    log_events=True,
    event_handlers=[events.append],
))

response = await agent.run("这个项目里有哪些文件？")

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

print(f"工具耗时: {tool_duration_ms:.1f}ms")
for event in failed_tools:
    print(f"失败工具: {event.name} ({event.error_kind})")
```

常用事件类型包括 `llm_start`、`llm_end`、`tool_start`、`tool_end`、`tool_error`、`retry` 和 `done`。工具错误会包含 `error_kind`，例如 `invalid_arguments`、`tool_not_found`、`timeout`、`permission_denied` 或 `tool_error`。

Plan-and-Execute tracing 还会输出 `plan_created`、`step_start`、`step_end`、`step_error` 和 `plan_revised`。Step 事件会把 step id 放在 `name` 中，并在 metadata 里包含 `step_index`、依赖、step 状态和 plan 状态等信息。

Skills tracing 还会输出：

- `skill_route_start`：包含 `metadata.candidate_names`、`metadata.candidate_count` 和 `metadata.router`
- `skill_route_end`：`content` 保存路由器的原始输出，另含 `metadata.matched_names`、`metadata.unrecognized_names` 和 `duration_ms`
- `skill_route_error`：`content` 为失败信息，另含 `metadata.error_type`、`metadata.fallback="no_skills"` 和 `duration_ms`
- `skill_injected`：包含被注入 Skill 的 `name`、`metadata.path` 和 `metadata.content_length`

`skill_route_end.content` 保存的是模型生成的完整 Router 输出，因此 tracing 日志可能包含敏感的提示词或路由数据；请相应配置日志、存储、访问权限和保留策略。

### 从 JSON 文件加载配置

```json
{
  "model": "gpt-4o",
  "system_prompt": "你是一个编程助手",
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

JSON 中 `mcp_servers` 根据 `command`（stdio）或 `url`（StreamableHTTP）自动识别类型。

### 从环境变量加载配置

```bash
export AIR_MODEL=gpt-4o
export AIR_SYSTEM_PROMPT="你是一个助手"
export AIR_MAX_ITERATIONS=30
export AIR_MCP_SERVERS='[{"command":"npx","args":["server"]}]'
```

```python
config = AgentConfig.from_env()          # 默认 AIR_ 前缀
config = AgentConfig.from_env(prefix="MYAPP_")  # 自定义前缀
agent = Agent(config)
```

支持的环境变量：

| 变量 | 类型 | 说明 |
| ---- | ---- | ---- |
| `AIR_MODEL` | str | 模型名称 |
| `AIR_API_KEY` | str | API 密钥（优先级高于 `OPENAI_API_KEY`） |
| `AIR_BASE_URL` | str | 自定义 API endpoint |
| `AIR_PROVIDER` | str | Provider 名称（支持 `openai`；不设置也使用 OpenAI） |
| `AIR_SYSTEM_PROMPT` | str | 系统提示词 |
| `AIR_MAX_ITERATIONS` | int | 最大工具调用轮次 |
| `AIR_STRATEGY` | str | Agent 策略：`react` 或 `plan_execute` |
| `AIR_MAX_PLAN_STEPS` | int | Plan-and-Execute 生成 plan 的最大 step 数 |
| `AIR_TOOL_TIMEOUT` | float | 工具调用超时（秒） |
| `AIR_MCP_SERVERS` | JSON | MCP server 列表 |
| `AIR_DEFAULT_HEADERS` | JSON | 自定义请求头 |
| `AIR_SKILLS_DIR` | str | Skills 文件目录路径 |
| `AIR_BUILTIN_TOOLS` | JSON | 内置工具配置 |
| `AIR_ENABLE_TRACING` | bool | 启用结构化事件分发 |
| `AIR_LOG_EVENTS` | bool | 以 JSON 形式记录结构化事件 |
| `AIR_MAX_TOOL_RETRIES` | int | 可重试工具错误的重试次数 |
| `AIR_MEMORY_ENABLED` | bool | 挂载 memory store 后启用 memory 检索 |
| `AIR_MEMORY_SEARCH_LIMIT` | int | 每次运行最多检索的 memory record 数 |
| `AIR_MEMORY_MAX_CHARS` | int | 每次运行注入的 memory 上下文最大字符数 |
| `AIR_MEMORY_SUMMARY_THRESHOLD` | int | 对话长度达到该阈值后才考虑自动写入 summary memory |

### 自定义 LLM Provider

OpenAI 仍然是默认 Provider。对于 OpenAI-compatible API，继续使用 `model`、`api_key`、`base_url` 和 `default_headers`：

```python
from air_agent import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="gpt-4o",
    api_key="sk-xxx",
    base_url="https://api.example.com/v1",
    default_headers={"X-API-Key": "custom-header"},
))
```

对于其他后端，可以传入实现了 `LLMProvider` 的对象。Provider 会返回中性的 `LLMResponse` 和 `LLMStreamChunk` 类型，因此可以把任何后端适配进来，而不依赖 OpenAI 专属 payload。

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

如果 `supports_tools = False`，一旦运行中有已注册或已启用的工具，就会明确失败，而不是静默忽略工具。内置工具默认启用，因此可以像上面示例那样关闭内置工具，或在 Provider 中实现工具支持。

### Skills（技能系统）

从技能文件夹目录加载技能指令。每个 Skill 是一个目录（kebab-case 命名），包含 `SKILL.md` 文件，使用 YAML frontmatter 定义元数据，Markdown 正文定义指令内容。

**目录结构：**

```text
skills/
├── brainstorming/
│   └── SKILL.md              # 必需：元数据 + 指令
├── data-analysis/
│   ├── SKILL.md
│   ├── scripts/              # 可选：可执行脚本
│   │   └── process_data.py
│   └── references/           # 可选：模板、Schema 等参考资料
│       └── data_schema.json
```

**SKILL.md 格式**（`skills/brainstorming/SKILL.md`）：

```markdown
---
name: brainstorming
description: 在进行创意工作或探索想法时使用
---

# 头脑风暴

每次只问一个问题来逐步细化想法。
```

**使用方式：**

```python
from air_agent import Agent, AgentConfig

config = AgentConfig(
    model="gpt-4o",
    skills_dir="./skills",  # 包含 skill 子目录的目录
)
agent = Agent(config)
response = await agent.run("我想头脑风暴一个新功能")
```

Skills 通过渐进式 Prompt 注入工作：

- 所有 Skill 元数据（名称 + 描述）始终包含在系统提示词中
- 当用户查询匹配到相关 Skill 时，完整 Skill 内容会被注入到对话上下文中
- Skill 匹配默认使用基于 LLM 的路由器；你可以提供自定义的 `SkillRouter` 实现

**自定义路由器：**

```python
from air_agent import SkillRouter

class KeywordRouter(SkillRouter):
    async def match(self, user_input: str, skills: list) -> list:
        return [s for s in skills if s.name in user_input.lower()]
```

### 内置工具

Agent 自带最小内置工具集，支持文件系统操作和 Shell 命令执行。默认启用，自动注册。

| 工具 | 说明 |
| ---- | ---- |
| `read_file` | 读取文件内容，支持 offset/limit |
| `write_file` | 写入文件，自动创建目录 |
| `list_directory` | 列出目录内容（含类型和大小） |
| `find_files` | 按 glob 模式查找文件 |
| `grep` | 正则搜索文件内容 |
| `run_shell` | 执行 Shell 命令 |

**默认使用（无需配置）：**

```python
from air_agent import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-4o", api_key="sk-xxx"))
# read_file, write_file, list_directory, find_files, grep, run_shell 均可用
```

**配置方式：**

```python
from air_agent import BuiltinToolsConfig

# 完全禁用内置工具
config = AgentConfig(model="gpt-4o", builtin_tools=BuiltinToolsConfig(enabled=False))

# 只启用部分工具
config = AgentConfig(model="gpt-4o",
    builtin_tools=BuiltinToolsConfig(tools=["read_file", "grep"]))

# 自定义沙箱和限制
config = AgentConfig(model="gpt-4o",
    builtin_tools=BuiltinToolsConfig(
        allowed_directories=["/project"],
        max_read_size=500_000,
        max_grep_results=50,
        default_timeout=60.0,
    ))
```

**安全机制：**

- **路径沙箱** — 文件工具只能访问 `allowed_directories` 内的路径（默认为当前工作目录）
- **命令黑名单** — 危险命令（`rm -rf /`、`sudo`、`mkfs` 等）被自动阻止
- **结果限制** — find/grep/list 结果数量和 shell 输出均可配置上限
- **截断通知** — 结果被截断时会告知 Agent，便于其调整查询策略

**`BuiltinToolsConfig` 配置项：**

| 字段 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `enabled` | bool | `True` | 总开关 |
| `tools` | list | `None` | 工具选择（`None` = 全部） |
| `allowed_directories` | list | `[]` | 沙箱目录（空 = 当前工作目录） |
| `max_read_size` | int | `1000000` | 文件读取大小上限（字节） |
| `default_timeout` | float | `30.0` | Shell 命令超时时间（秒） |
| `blocked_commands` | list | [...] | 被阻止的命令模式 |
| `max_find_results` | int | `200` | find 结果上限 |
| `max_grep_results` | int | `100` | grep 匹配上限 |
| `max_list_entries` | int | `500` | 目录列表条目上限 |
| `max_output_bytes` | int | `50000` | Shell 输出截断阈值 |

### 连接 MCP Server

```python
from air_agent import MCPServerStdio, MCPServerSSE

agent = Agent(AgentConfig(
    model="gpt-4o",
    mcp_servers=[
        MCPServerStdio(command="npx", args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]),
        MCPServerSSE(url="http://localhost:8080/mcp"),
    ],
))

async with agent:  # 自动连接/断开 MCP server
    response = await agent.run("列出 /tmp 下的文件")
```

支持 stdio 和 StreamableHTTP 两种 MCP transport。`MCPServerSSE` 是 URL 型 MCP server 的兼容命名。连接 MCP 后，server 暴露的工具会自动注册到 Agent 的工具列表中。

### 并行与角色化 Subagent

```python
from air_agent import SubagentConfig

results = await agent.delegate(
    tasks=[
        "分析 src/ 目录的代码结构",
        "检查 tests/ 的测试覆盖率",
        "生成 CHANGELOG",
    ],
    config=SubagentConfig(max_parallel=3, timeout=60),
)

for r in results:
    print(f"[{r.status}] {r.content[:100]}")
```

每个 task 会以独立 prompt 通过同一个 Agent 并发执行，并由 `SubagentConfig.max_parallel` 限制并发数。

`delegate()` 保持向后兼容：不传 roles 或 aggregation 时，仍然返回 `list[SubagentResult]`。如果需要协作式工作流，可以为 subagent 增加角色，每个角色都可以有自己的 prompt、工具、skills 目录和 memory scope。

```python
from air_agent import AgentRole

results = await agent.delegate(
    tasks=["审查实现", "检查安全风险"],
    roles=[
        AgentRole(name="reviewer", system_prompt="关注正确性和缺失测试。"),
        AgentRole(name="security", system_prompt="关注权限和数据暴露风险。"),
    ],
)

for result in results:
    print(result.role, result.status, result.content)
```

也可以让多个角色检查同一个任务：

```python
results = await agent.delegate(
    tasks=["评估这个发布计划"],
    roles=[
        AgentRole(name="product", system_prompt="评估用户价值。"),
        AgentRole(name="engineering", system_prompt="评估实现风险。"),
    ],
)
```

如果希望得到一个最终结果，而不是每个 subagent 一个结果，可以启用 aggregation：

```python
summary = await agent.delegate(
    tasks=["审查实现", "检查安全风险"],
    roles=[
        AgentRole(name="reviewer", system_prompt="关注正确性。"),
        AgentRole(name="security", system_prompt="关注安全。"),
    ],
    aggregation="summarize",  # "concat", "summarize", "vote"，或自定义 callable
)

print(summary.content)
```

每个 `SubagentResult` 都包含 `role`、`task`、`status`、`content`、`usage`、捕获到的 child `events` 和 `metadata`。

## 配置

```python
AgentConfig(
    model="gpt-4o",              # 模型名称
    api_key="sk-xxx",            # 或设置 OPENAI_API_KEY 环境变量
    base_url=None,               # 自定义 API endpoint
    provider=None,                # None/"openai" 或 LLMProvider 对象
    default_headers=None,         # 自定义 provider 请求头
    system_prompt="你是一个助手",  # 系统提示词
    memory=None,                  # MemoryStore 或 None
    memory_enabled=False,         # 启用 memory 检索
    memory_search_limit=5,        # 每次运行最多检索的 memory record 数
    memory_max_chars=4000,        # memory 上下文最大字符数
    memory_summary_threshold=12,  # 触发 summary memory 的对话轮次阈值
    strategy="react",             # "react" 或 "plan_execute"
    planner=None,                  # Planner 对象；仅支持程序化传入
    max_plan_steps=8,              # Plan-and-Execute step 上限
    max_iterations=20,           # 工具调用最大轮次
    tool_timeout=30.0,           # 单次工具调用超时（秒）
    mcp_servers=[],              # MCP server 列表
    skills_dir=None,             # Skills 文件目录路径
    builtin_tools=None,          # BuiltinToolsConfig 或 None 使用默认值
    enable_tracing=False,         # 输出结构化 RunEvent 记录
    log_events=False,             # 以 JSON 形式记录 RunEvent
    max_tool_retries=0,           # 可重试工具错误的重试次数
)
```

## 项目结构

```text
src/air_agent/
├── __init__.py          # 公开 API 导出
├── agent.py             # 核心 Agent（ReAct 循环 + 流式输出）
├── config.py            # 配置数据类
├── memory.py            # MemoryRecord、MemoryStore 与 memory store 实现
├── planner.py           # Planner 协议、LLMPlanner、Plan、PlanStep、StepResult
├── providers/
│   ├── types.py         # LLMProvider 协议 + 中立响应类型
│   └── openai.py        # 默认 OpenAI provider adapter
├── tracing.py           # RunEvent 分发与结构化事件日志
├── types.py             # Response, StreamEvent, AgentRole, SubagentResult
├── tools/
│   ├── base.py          # Tool 数据类
│   ├── registry.py      # 工具注册中心
│   └── builtin/
│       ├── config.py    # BuiltinToolsConfig
│       ├── _permissions.py  # 路径沙箱 + 命令黑名单
│       ├── file_tools.py    # 读写、列表、查找、grep
│       └── shell_tools.py   # run_shell
├── mcp/
│   ├── client.py        # MCP 客户端（stdio + streamable_http）
│   └── tool_adapter.py  # MCP tool → OpenAI 格式转换
├── skills/
│   ├── skill.py         # Skill 数据类 + SKILL.md 解析器
│   ├── manager.py       # SkillManager（目录扫描）
│   └── router.py        # SkillRouter 抽象类 + LLMSkillRouter
└── subagent.py          # 并行 subagent 管理器
```

## 依赖

- `openai` — LLM 调用与 tool calling
- `mcp` — MCP 协议客户端
- `pydantic` — 数据校验

## 开发

```bash
uv sync --group dev
uv run pytest tests/ -v
```

## License

MIT
