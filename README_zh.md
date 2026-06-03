# air-agent

[English](README.md)

轻量级 Python AI Agent 库。基于 OpenAI Chat Completions API，支持工具调用循环、MCP Server 连接、并行 Subagent 和流式输出。设计为可被其他 Python 项目直接引用。

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

### 基础对话

```python
import asyncio
from air_agent import Agent, AgentConfig

async def main():
    agent = Agent(AgentConfig(model="gpt-4o"))
    response = await agent.run("用一句话解释量子计算")
    print(response.content)

asyncio.run(main())
```

### 注册本地工具

```python
agent = Agent(AgentConfig(model="gpt-4o", api_key="sk-xxx"))

@agent.tool(name="add", description="计算两个数的和")
async def add(a: int, b: int) -> int:
    return a + b

response = await agent.run("3 加 5 等于多少？")
# Agent 会自动调用 add 工具并返回结果
```

参数类型从函数签名自动推导，生成 OpenAI tool calling 所需的 JSON Schema。

### 流式输出

```python
async for event in await agent.run("写一首关于编程的诗", stream=True):
    if event.type == "text":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[调用工具: {event.name}]")
    elif event.type == "tool_result":
        print(f"[工具结果: {event.content}]")
    elif event.type == "done":
        print(f"\n完成，token 用量: {event.usage}")
```

### Tracing 与结构化事件

Tracing 默认关闭。启用后，Agent 会为 LLM 调用、工具调用、重试、错误和完成状态输出结构化 `RunEvent` 记录。

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

### 多轮对话

```python
response = await agent.run("你好", conversation_id="session-1")
response = await agent.run("我刚才说了什么？", conversation_id="session-1")
# 第二轮会带上第一轮的上下文
```

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

JSON 中 `mcp_servers` 根据 `command`（stdio）或 `url`（SSE）自动识别类型。

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
| `AIR_SYSTEM_PROMPT` | str | 系统提示词 |
| `AIR_MAX_ITERATIONS` | int | 最大工具调用轮次 |
| `AIR_TOOL_TIMEOUT` | float | 工具调用超时（秒） |
| `AIR_MCP_SERVERS` | JSON | MCP server 列表 |
| `AIR_DEFAULT_HEADERS` | JSON | 自定义请求头 |
| `AIR_SKILLS_DIR` | str | Skills 文件目录路径 |

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

支持 stdio 和 StreamableHTTP 两种 MCP transport。连接 MCP 后，server 暴露的工具会自动注册到 Agent 的工具列表中。

### 并行 Subagent

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

每个 task 在独立的 Agent 实例中运行，互不干扰。

## 配置

```python
AgentConfig(
    model="gpt-4o",              # 模型名称
    api_key="sk-xxx",            # 或设置 OPENAI_API_KEY 环境变量
    base_url=None,               # 自定义 API endpoint
    system_prompt="你是一个助手",  # 系统提示词
    max_iterations=20,           # 工具调用最大轮次
    tool_timeout=30.0,           # 单次工具调用超时（秒）
    mcp_servers=[],              # MCP server 列表
    skills_dir=None,             # Skills 文件目录路径
    builtin_tools=None,          # BuiltinToolsConfig 或 None 使用默认值
)
```

## 项目结构

```text
src/air_agent/
├── __init__.py          # 公开 API 导出
├── agent.py             # 核心 Agent（ReAct 循环 + 流式输出）
├── config.py            # 配置数据类
├── types.py             # Response, StreamEvent, SubagentResult
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
