# air-agent 能力演进路线图

本文档根据 `docs/suggestion-from-deepseek.md` 整理 air-agent 接下来的能力演进方向。目标不是把 air-agent 扩展成重型平台，而是在保持轻量核心的前提下，为高级能力建立清晰、可插拔、可验证的演进路径。

## 项目定位

air-agent 是一个轻量级 Python AI Agent 库。当前 v0.2.0 已具备以下基础能力：

- 基于 OpenAI Chat Completions API 的 ReAct 工具调用循环。
- 多轮会话、流式输出、工具调用与工具结果事件。
- 本地工具注册、MCP 工具接入、内置文件与 Shell 工具。
- 路径沙箱、命令阻断、结果截断等基础安全机制。
- 并行 subagent 执行。
- Skills 目录结构、技能元数据注入、LLM 路由匹配。
- JSON 配置与 `AIR_` 环境变量配置。

后续演进应遵循 “Thin Harness, Fat Skills”：核心框架只负责稳定的运行循环、工具协议、配置、事件和扩展点；模型生态、记忆、多 Agent 协作、规划推理、观测与工具集成应优先以可选模块、插件或适配器形式接入。

## 演进原则

1. **保持轻量核心**
   - 核心依赖继续保持克制。
   - 新能力优先通过抽象接口和可选依赖实现。
   - 默认体验不能要求用户安装向量库、数据库、浏览器自动化或云服务 SDK。

2. **稳定公开 API**
   - 面向用户的 `Agent`、`AgentConfig`、工具注册、MCP 接入和 streaming 事件保持向后兼容。
   - 破坏性变化必须延后到明确版本窗口，并提供迁移说明。

3. **插件式重能力**
   - 记忆、模型 Provider、规划器、观测后端、工具包、协作协议都应能独立启用。
   - 内置实现只提供最小可用版本，高级实现交给扩展包。

4. **生产可观测**
   - Agent 的决策轨迹、模型调用、工具调用、错误、重试和 token 用量必须能被追踪。
   - 可观测性优先输出结构化事件，再适配日志、指标和追踪系统。

5. **安全默认值**
   - 文件、Shell、网络、MCP 和未来的动态插件能力必须默认收敛权限。
   - 高风险能力需要显式配置启用，并能审计调用记录。

## 总体优先级

建议优先级如下：

1. 可观测性与运行可靠性。
2. 多模型 Provider 抽象。
3. 记忆系统 MVP。
4. 规划执行与任务分解。
5. 多 Agent 协作。
6. 工具生态与插件注册。
7. 部署运维与 v1.0 兼容性承诺。

这个顺序与建议文档中的方向基本一致，但将可观测性提前。原因是后续所有复杂能力都会放大调试成本；先建立事件、追踪和错误恢复基础，可以让后续模块更容易测试、定位和演进。

## Milestone v0.3: 可观测性与运行可靠性

### 目标

让开发者能够理解一次 Agent 运行中发生了什么：模型调用了什么、工具为什么被调用、耗时和 token 用量如何、失败是否重试、最终结果来自哪些步骤。

### 范围

- 新增统一运行事件模型。
- 记录 LLM 调用、工具调用、工具结果、错误、重试、完成状态。
- 支持结构化日志输出。
- 为非 streaming 与 streaming 模式提供一致的可观测事件。
- 增加工具调用重试与超时分类。

### 非目标

- 不内置完整 UI 控制台。
- 不强依赖 OpenTelemetry、Prometheus 或外部日志系统。
- 不实现任务断点恢复，只为后续 checkpoint 能力预留事件边界。

### 规格要求

- 新增 `RunEvent` 或扩展 `StreamEvent`，至少包含：
  - `type`: `llm_start`、`llm_end`、`tool_start`、`tool_end`、`tool_error`、`retry`、`done`。
  - `run_id`、`conversation_id`、`iteration`、`timestamp`。
  - `name`、`arguments`、`content`、`duration_ms`、`usage` 等可选字段。
- `AgentConfig` 增加观测配置：
  - `enable_tracing: bool = False`
  - `log_events: bool = False`
  - `event_handlers: list[Callable] | None = None`，或等价的可插拔 handler 机制。
- 工具执行错误需要区分：
  - JSON 参数解析失败。
  - 工具不存在。
  - 工具超时。
  - 工具内部异常。
  - 权限或沙箱拒绝。
- 默认行为保持兼容：未启用 tracing 时，`Agent.run()` 的返回类型不变。
- 文档提供最小示例：打印运行轨迹、统计工具耗时、记录失败工具调用。

### 验收标准

- 单元测试覆盖事件字段、错误分类、工具超时、普通返回兼容性。
- streaming 与非 streaming 模式都能产生相同语义的事件。
- README 或 docs 提供一段可运行的 tracing 示例。

## Milestone v0.4: 多模型 Provider 抽象

### 目标

解除框架对 OpenAI 客户端实现的强绑定，让用户能在 OpenAI 兼容服务、Anthropic、本地模型或其他 Provider 之间切换，同时不破坏现有 OpenAI 用法。

### 范围

- 引入统一 LLM Provider 抽象。
- 默认 OpenAI Provider 继续支持现有行为。
- 支持 OpenAI-compatible base URL。
- 预留 Anthropic、本地模型、Ollama、vLLM 等 Provider 扩展点。
- 统一普通输出、streaming、tool calling、token usage 的最小接口。

### 非目标

- 不承诺一次性支持大量 Provider。
- 不在核心包内增加所有 Provider SDK。
- 不试图抹平所有模型能力差异，只定义框架需要的最小能力集。

### 规格要求

- 新增 `LLMProvider` 协议或抽象基类：
  - `complete(messages, tools=None, **options) -> LLMResponse`
  - `stream(messages, tools=None, **options) -> AsyncIterator[LLMStreamChunk]`
  - `supports_tools: bool`
  - `supports_streaming: bool`
- `AgentConfig` 支持：
  - `provider: str | LLMProvider | None = None`
  - `model: str`
  - `api_key`、`base_url`、`default_headers` 继续兼容。
- 内部 Agent loop 不直接依赖 `AsyncOpenAI` 的 response shape，而依赖统一响应类型。
- Provider 需要明确处理工具调用格式差异：
  - 支持原生 tool calling 的 Provider 走原生路径。
  - 不支持 tool calling 的 Provider 明确报错，或由未来 planner/adapter 处理。
- Skills 路由器也应复用 Provider 抽象，避免继续直接绑定 OpenAI client。

### 验收标准

- 现有 OpenAI 用法无需修改即可通过测试。
- 可通过 fake provider 测试 ReAct loop、streaming、tool call、skills routing。
- 文档说明如何实现自定义 Provider。

## Milestone v0.5: 记忆系统 MVP

### 目标

让 Agent 能在单次上下文窗口之外保留和检索有用信息，同时保持默认无状态、轻量、可关闭。

### 范围

- 引入记忆抽象接口。
- 提供内存版和文件版基础实现。
- 支持会话摘要、事实记忆、任务状态三类最小记忆。
- 在构建 messages 时支持检索并注入相关记忆。

### 非目标

- 不在核心中强依赖 Chroma、Pinecone、Postgres、Redis 等存储。
- 不默认启用长期记忆。
- 不实现复杂的自动遗忘策略，只提供可替换策略接口。

### 规格要求

- 新增 `MemoryStore` 协议：
  - `add(record: MemoryRecord) -> None`
  - `search(query: str, limit: int = 5) -> list[MemoryRecord]`
  - `summarize(conversation_id: str) -> str | None`
  - `clear(scope: str | None = None) -> None`
- `MemoryRecord` 至少包含：
  - `id`、`scope`、`kind`、`content`、`metadata`、`created_at`、`updated_at`。
- `AgentConfig` 支持：
  - `memory: MemoryStore | None = None`
  - `memory_enabled: bool = False`
  - `memory_search_limit: int = 5`
- 记忆注入应有清晰边界：
  - 以单独 system message 或上下文片段形式注入。
  - 标记来源，避免与用户输入混淆。
  - 注入内容有长度限制。
- 提供最小摘要策略：
  - 当会话历史超过阈值时生成摘要。
  - 摘要失败时不影响主流程。

### 验收标准

- 多轮对话可从 memory 中检索历史事实。
- 未启用 memory 时行为与 v0.2.0 兼容。
- 测试覆盖 memory 注入顺序、长度限制、检索为空、摘要失败。

## Milestone v0.6: 规划执行与任务分解

### 目标

在 ReAct 循环之外提供可选的 Plan-and-Execute 能力，使复杂任务能被拆分、执行、检查和修正。

### 范围

- 新增 Planner 抽象。
- 支持生成任务计划、执行步骤、收集结果、最终汇总。
- 支持步骤级事件与错误处理。
- 为未来 checkpoint 和恢复能力预留结构。

### 非目标

- 不替代默认 ReAct loop。
- 不内置复杂的思维树、思维图实现。
- 不保证所有任务都自动规划；用户应显式启用。

### 规格要求

- 新增 `Planner` 协议：
  - `create_plan(goal: str, context: PlanContext) -> Plan`
  - `execute_step(step: PlanStep, context: PlanContext) -> StepResult`
  - `revise_plan(plan: Plan, result: StepResult) -> Plan`
- `Agent.run()` 可通过参数或配置启用：
  - `strategy="react" | "plan_execute"`，默认 `react`。
- `Plan` 至少包含：
  - `goal`、`steps`、`status`、`created_at`。
- `PlanStep` 至少包含：
  - `id`、`description`、`status`、`dependencies`。
- 规划执行过程必须产生可观测事件：
  - `plan_created`、`step_start`、`step_end`、`step_error`、`plan_revised`。

### 验收标准

- 简单多步骤任务能生成计划并顺序执行。
- 单步失败可被记录，且最终结果能说明失败步骤。
- 默认 ReAct 路径不受影响。

## Milestone v0.7: 多 Agent 协作

### 目标

把当前简单并行 subagent 扩展为可组合的多 Agent 协作机制，支持角色、独立上下文、结果汇总和基础仲裁。

### 范围

- 支持角色化 Agent 配置。
- 支持任务分配、并行执行、结果汇总。
- 支持基础仲裁策略：主 Agent 汇总、投票、人工选择。
- 保持现有 `delegate()` 的简单用法。

### 非目标

- 不立即实现完整 A2A 协议。
- 不提供分布式 Agent 网络。
- 不在核心里内置复杂辩论系统。

### 规格要求

- 新增 `AgentRole` 或等价配置：
  - `name`、`description`、`system_prompt`、`tools`、`skills_dir`、`memory_scope`。
- `delegate()` 支持：
  - `tasks: list[str]`
  - `roles: list[AgentRole] | None`
  - `aggregation: "concat" | "summarize" | "vote" | Callable`
- 每个 subagent 必须具备独立：
  - message history。
  - skill matching。
  - memory scope。
  - tracing context。
- 主 Agent 可接收结构化 `SubagentResult`：
  - `role`、`task`、`status`、`content`、`usage`、`events`。

### 验收标准

- 原有 `agent.delegate(tasks)` 用法继续可用。
- 可为不同任务指定不同角色 prompt。
- 多个 subagent 失败时主流程仍能返回结构化结果。

## Milestone v0.8: 工具生态与插件注册

### 目标

让工具、技能、Provider、Memory、Planner 等扩展可以被统一发现、声明和加载，逐步形成可复用生态。

### 范围

- 定义插件清单格式。
- 支持本地插件目录加载。
- 支持工具包分组注册。
- 支持插件能力声明与权限提示。
- 为未来远程注册中心预留字段。

### 非目标

- 不立即建设中心化市场。
- 不默认执行来自网络的动态代码。
- 不绕过现有工具安全边界。

### 规格要求

- 插件清单建议命名为 `air-agent-plugin.json`，至少包含：
  - `name`、`version`、`description`、`entrypoint`。
  - `capabilities`: `tools`、`skills`、`provider`、`memory`、`planner`。
  - `permissions`: 文件、Shell、网络、环境变量等声明。
- `AgentConfig` 支持：
  - `plugins: list[str] = field(default_factory=list)`
  - `plugin_permissions: dict[str, Any] | None = None`
- 插件加载必须：
  - 不影响未启用插件的启动速度。
  - 明确报错插件加载失败原因。
  - 不吞掉权限拒绝错误。
- 工具包注册应支持命名空间，避免工具名冲突：
  - 例如 `web.search`、`data.read_csv`。

### 验收标准

- 本地插件可注册一组工具并被 Agent 调用。
- 插件权限声明能在文档和错误信息中体现。
- 工具名冲突有确定行为：拒绝注册或显式覆盖。

## Milestone v1.0: 生产部署与兼容性承诺

### 目标

将 air-agent 收敛为稳定、可依赖的 1.0 版本：核心 API 稳定，扩展机制清晰，测试覆盖关键路径，文档能支撑真实项目集成。

### 范围

- 明确 semver 兼容策略。
- 完善错误类型和异常层级。
- 提供生产配置示例。
- 提供 Docker 示例和部署建议。
- 完成核心模块文档与迁移指南。

### 非目标

- 不要求 air-agent 自带服务端平台。
- 不提供 Kubernetes operator 或托管服务。
- 不内置任务队列系统，但文档说明如何集成外部队列。

### 规格要求

- 公共 API 分类：
  - Stable: `Agent`、`AgentConfig`、工具注册、MCP 配置、Provider、Memory、Planner 基础协议。
  - Experimental: 插件注册中心、多 Agent 高级仲裁、checkpoint。
- 错误体系至少包含：
  - `AirAgentError`
  - `ProviderError`
  - `ToolExecutionError`
  - `ToolPermissionError`
  - `MCPConnectionError`
  - `PluginLoadError`
  - `MemoryError`
- 文档必须覆盖：
  - 快速开始。
  - Provider 自定义。
  - 工具安全配置。
  - tracing 示例。
  - memory 示例。
  - plugin 示例。
  - 从 v0.x 迁移到 v1.0。

### 验收标准

- 核心测试覆盖率达到项目约定阈值。
- 公开 API 有稳定性说明。
- 所有 README 示例通过测试或文档测试验证。
- v1.0 发布说明列出兼容性承诺和实验性 API。

## 横向规格要求

### 配置

- 所有新能力都应支持 Python 对象配置。
- 对用户常用能力提供 JSON 和环境变量配置。
- 高风险能力必须显式启用。
- 可选依赖缺失时，错误信息应说明安装方式或替代方案。

### 测试

- 新抽象必须有 fake implementation，避免测试依赖外部服务。
- 每个里程碑至少覆盖：
  - 正常路径。
  - 配置关闭路径。
  - 失败路径。
  - 与现有 API 的兼容路径。
- 网络、模型、数据库相关测试默认 mock 或 fake。

### 文档

- 每个新能力必须包含：
  - 最小可运行示例。
  - 配置说明。
  - 安全注意事项。
  - 与现有能力的组合方式。
- README 只保留最常用路径，复杂内容放入 `docs/`。

### 安全

- 新增网络、插件、Shell、文件、外部服务能力时必须说明权限边界。
- 工具调用参数应可记录、可截断、可隐藏敏感字段。
- 错误信息不能泄露 API key、环境变量或敏感 header。

## 暂缓事项

以下能力有价值，但不建议在 v1.0 前作为核心目标：

- 完整中心化插件市场。
- 内置 Web UI 调试控制台。
- 完整 A2A 协议实现。
- 分布式 Agent 集群。
- 内置向量数据库。
- 内置任务队列。
- 复杂 Tree-of-Thought 或 Graph-of-Thought 框架。

这些能力可以通过实验性扩展包探索，成熟后再考虑是否进入核心协议层。

## 建议的下一步

1. 先实现 v0.3 可观测性与可靠性，因为它会降低后续所有复杂能力的调试成本。
2. 在 v0.3 中确定稳定的事件模型，后续 Provider、Memory、Planner、多 Agent 都复用同一事件语义。
3. v0.4 前先把现有 OpenAI 调用封装到默认 Provider 中，确保后续模型生态扩展不会侵入 Agent loop。
4. 每个里程碑完成后更新本文档，将已完成内容移动到 changelog 或发布说明中。
