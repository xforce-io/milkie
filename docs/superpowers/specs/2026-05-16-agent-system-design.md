# milkie — Agent System Architecture Design

Date: 2026-05-16  
Status: Draft  
Language: TypeScript

---

## Overview

milkie 是一个 TypeScript Agent 框架，目标是同时作为底层框架和上层产品使用。核心场景是多 Agent 协作（中心化编排为主，底层 Agent 保留自主空间）。

**核心命题**：Agent 的本质是 FSM（有限状态机/Statechart）。所有 Agent 模式都是同一个 FSM Runtime 的不同配置，LLM continuous 对话是 FSM 的退化形式。

**Goal 是 Agent run 的顶层语义单元。** 一次 Agent 运行由一个 goal 驱动，goal 是不可变的运行输入，贯穿整个执行过程：写入 `agent.run` span 作为可比较的基准，供 Experiment 做跨版本效果对比。Goal ≠ current_turn：goal 是调用方的意图声明，current_turn 是执行过程中的动态输入。

---

## 1. 核心抽象：Agent = FSM

### 统一执行模型

FSM State 只有两种类型，所有 Agent 模式都是其组合：

```
type: action  — 不调用 LLM，直接执行（spawn sub-agent / run handler）
type: llm     — LLM loop，由执行结果决定何时退出
```

**`type: llm` 统一退出规则：**

| LLM 输出 | 结果 |
|---------|------|
| tool call → tool emits FSM event | 立即退出当前 state，按 `on:` 声明的转移执行 |
| tool call → tool 返回数据（无 event） | 追加 observation，继续 loop |
| 纯文字输出 | emit DONE；有 `on.DONE:` 则转移，无则等待用户 |

**常见 Agent 模式由此自然涌现，无需独立模板类型：**

- **ReAct**：单一 `type: llm` state，工具只返回数据 → LLM 多轮 loop 直到输出文字
- **意图路由**：`type: llm` + 单工具（emit 事件）→ 单次 LLM call 后路由到目标 state
- **对话等待**：`type: llm` 输出文字，无 `on.DONE` 出口 → 等待用户下一条消息

### Cognitive Tools 与 FSM 的关系

Cognitive tools（`think` / `create_plan` / `update_step`）是 FSM 的**软实现**：推理链隐式嵌在 tool 的 schema 与描述中，由 LLM 的 token 概率驱动，而非代码逻辑。

两种方式可以共存：
- **硬转移**：代码条件强制（关键业务逻辑）
- **软转移**：Cognitive tools 引导（推理过程，LLM 自主）
- **事件转移**：外部信号触发（用户输入、Workflow 回调）

---

## 2. 内部分层

每个 Agent 实例包含两个正交的层，互不共享：

```
┌─────────────────────────────────────────────┐
│              Agent Runtime                  │
│   事件循环 · 生命周期 · 并发 · 错误恢复       │
├──────────────────────┬──────────────────────┤
│     FSM Layer        │   LLM Context Layer  │
│  ─────────────────   │  ──────────────────  │
│  当前状态             │  消息历史（连续）      │
│  转移规则             │  System prompt       │
│  状态数据             │  Base tools          │
│  硬/软/事件转移       │  历史压缩策略         │
│                      │  工作记忆             │
└──────────────────────┴──────────────────────┘
```

**FSM Layer** 是 driver，决定何时调用 LLM、解析输出、驱动状态转移。  
**Context Layer** 是 service，被动响应 FSM 调用，提供 LLM 请求构建和对话历史管理能力。  
FSM 永远主动，Context 永远响应。

### Agent 身份原则

**System prompt 和 base tools 在 Agent 创建时固定，整个生命周期不变。** Context 是连续的对话历史，FSM 只能向其中追加内容（observations、messages），不能改变 Agent 身份。

如果某个场景需要完全不同的 system prompt，正确做法是通过 `Task` tool spawn 一个新的 sub-agent，而非在当前 Agent 内切换身份。

```typescript
interface AgentConfig {
  version:      string           // semver，任何依赖变化即产生新版本
  systemPrompt: string           // 创建时固定，生命周期内不变
  fsm:          FSMDefinition    // FSM 状态机定义
  toolboxes?:   Record<string, string>   // name → pinned semver；tool 由 toolbox 提供
  skills?:      Record<string, string>   // name → pinned semver；skill_request 激活，不触发版本解析
  subAgents?:   Record<string, string>   // agentId → pinned semver；Runtime 自动生成具名 tool
  model: {
    provider:  string
    model:     string
    adapter:   string
    baseUrl?:  string
    options?:  Record<string, unknown>
  }
  stateStore?:  'memory' | 'sqlite' | 'redis'  // 默认 memory
  dispatch?:    'local' | 'queue'              // sub-agent 执行位置，默认 local；queue 为 v2
}

interface FSMDefinition {
  states: FSMState[]
}

interface FSMState {
  name:            string
  type:            'llm' | 'action'
  instructions?:   string                  // type: llm，注入 instructions bucket
  tools?:          string[]                // 该 state 可用工具；省略则继承 agent 全部工具
  on?:             Record<string, string>  // event → target state；内置事件：DONE / ERROR / INTERRUPT
  handler?:        string                  // type: action，执行的 handler 名称
  terminal?:       boolean                 // 终止状态，无出口
  max_iterations?: number                  // type: llm，最大 LLM loop 次数（安全上限）
}
```

### Context Layer：多 Bucket 结构

Context 不是 flat message array，而是按稳定性分层的多个 bucket，顺序设计同时服务于 **LLM 理解质量** 和 **prefix cache 命中率**：

```
[CACHED PREFIX ─ 跨 turn 稳定]
┌──────────────────────────────────────┐
│  system_prompt          ← breakpoint 1 │  固定，永不改变
├──────────────────────────────────────┤
│  instructions           ← breakpoint 2 │  已加载 Skills 的 instructions
└──────────────────────────────────────┘

[UNCACHED ─ 每 turn 变化]
┌──────────────────────────────────────┐
│  history                              │  对话历史，滑动窗口或摘要
├──────────────────────────────────────┤
│  working_memory                       │  当前 turn 中间状态，激进压缩
├──────────────────────────────────────┤
│  current_turn                         │  当前输入，永不压缩，永远靠近末尾
└──────────────────────────────────────┘
```

各 bucket 属性：

| Bucket | 可变性 | 压缩策略 | Cache |
|--------|--------|---------|-------|
| system_prompt | 不可变 | 无 | breakpoint 1 |
| instructions | Skill 驱动 | 低频摘要 | breakpoint 2 |
| history | 累积 | 滑动窗口 / 摘要 | 无 |
| working_memory | FSM 追加 | 激进压缩 | 无 |
| current_turn | 每 turn 替换 | 无 | 无 |

**Prefix cache 约束：**
- Skill 加载/卸载使 breakpoint 2 失效，成本上升；instructions 在单个 turn 执行过程中冻结，只能在 turn 开始前或 resume 后重建
- working_memory 必须在 cache breakpoint 之后，否则每步变化会持续 invalidate cache
- 压缩优先级：working_memory → history → instructions（cached 区域最后动）

**Context epoch：**

每次 instructions 集合变化都会产生新的 `contextEpoch`。同一个 epoch 内 prefix cache 可复用；epoch 变化后重新构建 cached prefix，并把新的 loaded skill snapshot 写入 checkpoint。

```typescript
interface IContextLayer {
  // 各 bucket 独立操作
  setCurrentTurn(input: Input): void
  appendHistory(message: Message): void
  appendWorkingMemory(obs: Observation): void
  loadInstructions(skill: Skill): void
  unloadInstructions(skillName: string): void

  // 统一出口：内部处理 token budget、压缩、cache_control 标记
  buildRequest(): LLMRequest
}
```

`buildRequest()` 负责注入 `cache_control` breakpoint 标记，FSM 完全不感知 cache 细节。

### 单 Agent 事件循环

```
输入事件
  ↓
FSM: 当前状态 + 转移判断
  ↓
Context: 组装 LLM messages
  ↓
Model Gateway: 调用底层模型 provider
  ↓
解析响应
  ├─ tool_use × 1  → 串行执行
  └─ tool_use × N  → 并发执行 + allSettled join
  ↓
Observation → Context
  ↓
FSM 转移 → 循环 / 终止
```

---

## 3. Model Gateway（模型连接层）

Model Gateway 是 Agent Runtime 与底层模型 provider 之间的薄适配层。它只负责协议转换、stream 事件归一化、provider-specific 参数透传和用量统计，不拥有 Agent loop、Context、Tool 执行或 checkpoint 语义。

### 设计原则

- **Runtime from scratch**：FSM、事件循环、tool join、中断、checkpoint/resume、Trajectory 由 milkie 自己实现
- **Provider adapter 可复用库**：底层可以使用官方 SDK，或类似 `pi-ai` / `pi-agent` 的低层模型连接能力，但只能封装在 adapter 内
- **第三方 Agent loop 不进入核心路径**：不直接采用外部 Agent Runtime，避免其上下文管理、tool loop、事件调度与 milkie 的 FSM Runtime 冲突
- **请求/响应类型由 milkie 定义**：第三方库类型不得泄漏到 FSM、Context、Tool Registry、Trajectory Store

### 接口

```typescript
interface IModelGateway {
  complete(request: ModelRequest): Promise<ModelResponse>
  stream(request: ModelRequest): AsyncIterable<ModelEvent>
}

interface ModelRequest {
  model:          string
  messages:       Message[]
  tools?:         ToolSchema[]
  toolChoice?:    unknown
  responseFormat?: unknown
  cacheControl?:  CacheControl[]
  reasoning?:     ReasoningOptions
  metadata?:      Record<string, unknown>
}

type ModelEvent =
  | { type: 'message_delta'; data: { text: string } }
  | { type: 'tool_call_start'; data: { toolCallId: string; name: string } }
  | { type: 'tool_call_delta'; data: { toolCallId: string; delta: unknown } }
  | { type: 'tool_call_done'; data: { toolCallId: string; input: unknown } }
  | { type: 'usage'; data: ModelUsage }
  | { type: 'error'; data: { code: string; message: string; retryable?: boolean } }

interface ModelResponse {
  content:       MessageContent[]
  toolCalls:     ToolCall[]
  usage?:        ModelUsage
  finishReason?: string
  raw?:          unknown        // 仅供 debug/trajectory，不进入 FSM 判断
}
```

### Adapter 结构

```
Agent Runtime / FSM / Context
          ↓
      Model Gateway
          ↓
OpenAIAdapter / AnthropicAdapter / GeminiAdapter / OpenRouterAdapter / LocalAdapter
          ↓
官方 SDK / OpenAI-compatible endpoint / 第三方低层模型库
```

Provider adapter 必须保证：
- 不修改 `messages` 顺序，不私自注入 system prompt，不重写 tools schema
- 保留 provider 原始错误码、finish reason、usage 和 request id，供 `llm.call` span 记录
- 支持 provider-specific 参数透传，但透传字段必须留在 `metadata` 或 adapter config 中
- stream 事件必须归一化为 `ModelEvent`，Runtime 不感知 provider delta 格式
- cache control、reasoning、response format 等高级能力若 provider 不支持，adapter 必须显式降级或报错

### 第三方库使用边界

允许使用类似 `pi-ai` 的库来减少多 provider 连接成本，但使用位置限制在 provider adapter 内：

```
允许：
milkie OpenAIAdapter → pi-ai/openai sdk → provider

不允许：
milkie Agent Runtime → external agent loop → provider
```

引入第三方模型库前必须满足：
- 暴露低层 request/response 或 stream API，而不是只暴露 agent session
- 不接管 tool execution、message compression、memory 或 retry policy
- 能暴露原始 provider response，便于 trajectory、debug 和版本归因
- 版本可 pin，并写入 `resolvedManifest`

### 配置示例

```yaml
model:
  provider: openrouter
  model: anthropic/claude-sonnet-4.5
  adapter: openai-compatible
  baseUrl: https://openrouter.ai/api/v1
  reasoning:
    effort: medium
```

Model Gateway 是可替换组件。改变 adapter 实现、provider、model 或关键参数，都应进入 AgentConfig 版本 diff，并写入 Trajectory 的 resolved manifest。

---

## 4. 多 Agent 架构

### 独立性原则

每个 Agent 实例拥有独立的 FSM + 独立的 LLM Context，实例间不共享 context。父 Agent 通过 Task tool prompt 显式传递必要信息，子 Agent 结果作为 observation 回到父 Agent context。

### 并行执行（两层）

**Intra-agent 并行**（单 Agent 内）：

LLM 在单次响应中输出多个 `tool_use` block，Runtime 可并发执行，但 join 语义必须是 `allSettled`：所有 tool 都进入终态后，Runtime 生成一组结构化 observation，再继续 FSM 转移。

并发工具执行规则：
- 任一 tool 失败不会抢占取消其他已启动 tool；失败作为 observation 交给 FSM 决定重试、降级或进入 error handling
- 有副作用的 tool 必须支持幂等键（`toolCallId` / `agentRunId` / `stepId`），便于 resume 后去重
- 不声明为 `parallelSafe` 的 tool 不参与同批并发，Runtime 按顺序执行
- interrupt 只在本批 tool 全部 settled 后处理，保证 checkpoint 不记录半完成状态

**Inter-agent 并行**（跨 Agent）：

Sub-agent 以**具名 tool** 的形式暴露给 Orchestrator LLM，而非通过 generic `Task` tool。Framework 在 Tool Registry 初始化时，把 `AgentConfig.subAgents` 里声明的每个子 Agent 自动生成对应 tool schema（name = agentId，description 来自子 AgentConfig system prompt 摘要），LLM 通过标准 tool_use 机制调用。

Framework 拦截子 Agent tool 调用，内部转为 `AgentInvokeRequest`，并发启动子 Agent 实例（独立 FSM + Context），用 `allSettled` join 等待全部结束，将结构化结果作为 observations 返回给父 Agent。

```
Orchestrator Agent
  └─ researcher(...) × 1 + coder(...) × 1 → allSettled join
       ↓ Framework 拦截，转为 AgentInvokeRequest
  Researcher Agent    Coder Agent
  (独立 FSM+Context)  (独立 FSM+Context)
       ↓ 返回 TaskResult
  Orchestrator ← observation
```

**`Task` tool 是框架内部实现机制，不直接暴露给 LLM。** 子 Agent 调用的唯一入口是通过具名 tool，以保持 LLM 侧的 schema 引导和能力自描述。

### TaskResult 类型

```typescript
type TaskResult =
  | { status: 'success';     result: string }
  | { status: 'error';       reason: string; retryable?: boolean }
  | { status: 'interrupted'; checkpointId: string }
```

`interrupted` 不是错误，父 Agent 不触发 error handling，保存 checkpointId 供 resume 使用。

---

## 5. Tool Registry

### Tool 统一模型（概念层）

所有 tool 在 Runtime 层是同一种结构，没有"类型"区分：

```typescript
interface ToolDefinition {
  name:          string
  description:   string
  inputSchema:   JSONSchema
  handler:       (input: unknown, ctx: ToolContext) => Promise<unknown>
  parallelSafe?: boolean   // 默认 false；true 时允许在同批 allSettled 中并发执行
}

interface ToolContext {
  workingMemory: WorkingMemory   // 读写当前 turn 的中间状态
  agentFactory:  AgentFactory    // spawn 独立 Agent 实例
  stateStore:    IStateStore     // 持久化读写
  emit:          (event: string, payload?: unknown) => void  // 触发 FSM 状态转移
}
```

Runtime 对所有 tool 一视同仁：收到 `tool_use` → 找 handler → 执行 → 返回结果。差别只在 handler 内部使用了哪些 `ctx` 服务：

```
Tool (ToolDefinition)
├── Intra-agent tool          — handler 读写 ctx.workingMemory
│   └── Cognitive toolbox     — think / create_plan / update_step
│
├── Inter-agent tool          — handler 调 ctx.agentFactory
│   ├── Task tool             — 一个 entry，参数决定实例化哪个 AgentConfig
│   └── Named sub-agent tool  — AgentConfig.subAgents 自动生成的具名 wrapper，handler 同上
│
└── Domain tool               — handler 不依赖平台 ctx 服务
    └── search / filesystem / http / ...
```

**Intra-agent vs Inter-agent** 是最核心的分界线：
- **Intra-agent tool**：在当前 agent 的 FSM cycle 内执行，写到当前 agent 的 `workingMemory`，对当前 agent 下一轮推理立即可见
- **Inter-agent tool**：跨越 agent 边界，启动独立的 FSM + Context；子 agent 内部状态对父完全不可见，**子 agent 无法写到父 agent 的 workingMemory**

**Cognitive toolbox 的 stateless handler / stateful ctx**：handler 本身是纯函数（stateless），状态全部住在 `ctx.workingMemory`（属于 Context Layer，随 checkpoint 持久化）。典型场景：`create_plan` 写入计划步骤，`update_step` 逐一划掉，LLM 每轮都能读到当前执行进度，resume 后计划状态完整恢复。

**"Generic" 是设计模式，不是类型层级**：Task tool 的 "generic" 指一个 entry 参数化选择 AgentConfig；Named sub-agent tool 是 Task 的具名 wrapper，两者同属 Inter-agent tool 类别。

### Task Tool：泛型单入口

Tool Registry 里只有**一个** `Task` tool。handler 使用 `ctx.agentFactory` spawn 独立 Agent 实例，等待其运行结束后把输出作为 tool result 返回：

```typescript
registerTool({
  name: 'Task',
  schema: { agentId: string; goal: string; input: unknown },
  handler: async ({ agentId, goal, input }, ctx) => {
    const config = await ctx.agentRegistry.get(agentId)
    const child  = ctx.agentFactory.spawn({ config, goal, input })
    const result = await child.run()
    return result.output
  }
})
```

与普通 tool 的唯一区别：handler 执行时会启动一个完整的 Agent 实例（独立 FSM + Context），而不是调用一个函数。

### Sub-agent Tools：具名 wrapper（自动生成）

直接暴露泛型 `Task` tool 时，LLM 需要知道合法的 `agentId` 列表，schema 引导弱。因此 Runtime 在初始化时，把 `AgentConfig.subAgents` 里声明的每个子 Agent 自动生成一个**具名 tool**，底层路由到同一套 Task 逻辑：

```typescript
// Runtime 初始化时自动生成，非手写
for (const [agentId, version] of Object.entries(config.subAgents)) {
  registerTool({
    name:        agentId,
    description: registry.getDescription(agentId, version),  // 来自子 AgentConfig system prompt 摘要
    inputSchema: registry.getInputSchema(agentId, version),
    handler: (input, ctx) =>
      taskHandler({ agentId, goal: input.goal, input }, ctx)  // 路由到同一个 Task handler
  })
}
```

LLM 看到结构化的具名 tool（更好的 schema 引导和能力自描述），Runtime 执行的是同一套 spawn 路径。`Task` tool 是实现机制，具名 tool 是 LLM 侧的接口形态。

### Cognitive Toolbox

`think` / `create_plan` / `update_step` 不是平台硬编码的特殊类别，而是一个普通 **toolbox**（`cognitive: "1.0.0"`），handler 读写 `ctx.workingMemory`。声明方式与其他 toolbox 相同：

```yaml
toolboxes:
  cognitive: "1.0.0"   # 提供 think / create_plan / update_step
  search:    "1.0.0"
```

```typescript
// cognitive toolbox 内部实现示例
registerTool({
  name: 'think',
  description: 'Think step-by-step before acting. Use freely — has no side effects.',
  inputSchema: { thoughts: string },
  handler: async ({ thoughts }, ctx) => {
    ctx.workingMemory.append({ type: 'thought', content: thoughts })
    return { recorded: true }
  }
})

registerTool({
  name: 'create_plan',
  description: 'Create a checklist of steps. Call once at the start of a multi-step task.',
  inputSchema: { steps: string[] },
  handler: async ({ steps }, ctx) => {
    const plan = { id: uuid(), steps: steps.map((s, i) => ({ id: i, desc: s, status: 'pending' })) }
    ctx.workingMemory.set('plan', plan)
    return plan
  }
})

registerTool({
  name: 'update_step',
  description: 'Mark a step as done or failed. If failed, revise the plan with create_plan.',
  inputSchema: { stepId: number, status: 'done' | 'failed' },
  handler: async ({ stepId, status }, ctx) => {
    const plan = ctx.workingMemory.get('plan')
    plan.steps[stepId].status = status
    ctx.workingMemory.set('plan', plan)
    return plan.steps
  }
})
```

Plan 状态存在 `workingMemory` 中，随 checkpoint 持久化，resume 时自动恢复。

### Tool Registry 内容来源

运行时激活的 tools 来自三个来源：

| 来源 | 说明 |
|------|------|
| 系统内置 | `Task`、`skill_list`、`skill_request`；平台注册，始终可用 |
| Toolbox | 相关 tools 的打包单元；声明在 `AgentConfig.toolboxes`，Runtime 初始化时加载（含 cognitive toolbox）|
| Sub-agent tools | `AgentConfig.subAgents` 声明的子 Agent，Runtime 初始化时自动生成具名 tool |

**Toolbox** 是 tools 的分发和组织方式，不是独立的运行时组件。

### 配置驱动原则

Tool Registry 的内容由**配置**决定，配置是唯一来源。CLI 是操作配置的人机接口，不是独立的注册路径。

```
CLI / 编辑器
  └─ 操作 ──→ Agent 配置文件（MD with YAML frontmatter）
                  └─ Runtime 读取 ──→ AgentConfig ──→ Tool Registry 初始化
```

Agent 配置文件格式（MD with YAML frontmatter）：

```markdown
---
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      tools: [browser, code]
toolboxes:
  browser:    "2.0.0"
  code:       "1.0.0"
skills:
  research:   "1.1.0"
sub_agents:
  researcher: "1.0.0"
  coder:      "2.1.0"
model:
  provider: openrouter
  model: anthropic/claude-sonnet-4.5
  adapter: openai-compatible
  baseUrl: https://openrouter.ai/api/v1
state_store: redis
---

You are a research assistant specialized in...
（正文为 system prompt）
```

CLI 提供组装能力：

```bash
milkie agent create --config agent.md
milkie agent run <agent-id>
milkie agent resume <checkpoint-id>
milkie toolbox install browser    # 更新配置
milkie skill add research         # 更新配置
```

---

## 6. Skill Registry（Resource 层）

Skill 是 **Resource**，不是 Tool：
- Tool 被调用（动词），有副作用
- Skill 被读取（名词），加载后内容注入 LLM Context，改变行为

### 加载机制（系统 Tools）

```
skill_list()        → 返回所有可用 skill 名称 + 摘要
skill_request(name) → 请求下个 context epoch 加载 skill
```

Skill 加载采用 **epoch 边界生效**：LLM 可以在当前 turn 中请求 skill，但 Runtime 不立即修改当前 LLM request 的 instructions bucket，而是在本轮结束、下一轮开始或 resume 重建 context 时应用。这样保留运行期按需发现能力，同时不破坏单 turn 内 prefix cache 和 Agent 身份稳定性。

### Skill 与 Toolbox 的关系

Skill 定义中可以声明它依赖哪些 Toolbox（`requires_toolboxes`）。但 Toolbox 的加载时机不是 skill request，而是 **Agent 初始化时**，从 AgentConfig 的 `dependencies.toolboxes` 统一加载进 Tool Registry。

规则：
- `skill_request` 只改变下个 context epoch 的 instructions，不修改 Tool Registry
- Skill 引用的 Toolbox 必须已在 AgentConfig 的 `toolboxes` 中声明
- 配置构建时（`milkie skill add`）自动检查并补全缺失的 toolbox 依赖
- 运行时 Tool Registry 在初始化后保持稳定，与 Agent 身份不变原则一致

```yaml
# skill 定义示例（skill.yaml）
name: research
version: "1.1.0"
requires_toolboxes:
  - browser
  - search
instructions: |
  You are a research specialist. Use the browser and search tools to...
```

### Skill Registry 内容

- **Skill 定义**：instructions + requires_toolboxes

### Registry 解析（来源与优先级）

Skills 和 Toolboxes 的来源按优先级：

```
1. 本地路径   ./skills/research/     开发、自定义
2. 工作区     ~/.milkie/registry/    本地缓存、全局安装
```

版本解析流程：

```
AgentConfig 声明 skill:research@1.1.0
  └─ 查本地路径 → 命中则直接用
  └─ 查工作区缓存 → 命中则直接用
  └─ 构建时锁定 resolved SHA（写入 resolved manifest）
```

`milkie skill add research@1.1.0` 从本地或工作区预装依赖，运行时从缓存加载。远端 Registry 为 v2 特性。

### Agent Template 表达形式

**原则：结构声明式，逻辑代码化。**

FSM 拓扑（状态、转移）用声明式 YAML 表达，可序列化、可可视化、可存配置文件。Tool handler 中的 `ctx.emit()` 触发 FSM 事件，完成状态转移。

#### State 定义

```yaml
fsm:
  states:
    - name: <state-name>
      type: llm | action         # 执行类型（必填）
      instructions?: string      # 注入 instructions bucket 的 LLM 指令（type: llm）
      tools?: string[]           # 该 state 可用工具；省略则继承 agent 全部工具
      on:                        # FSM 转移规则
        <EVENT>: <target-state>  # tool emit 的自定义事件
        DONE: <target-state>     # LLM 纯文字输出时自动触发
      handler?: string           # type: action 时执行的 handler 名称
      terminal?: boolean         # 终止状态（无出口）
```

#### 工具触发 FSM 事件

Tool handler 通过 `ctx.emit()` 触发 FSM 状态转移：

```typescript
handler: async (input, ctx) => {
  // 确定性逻辑...
  ctx.emit('BOOKING_READY', payload)  // 触发 FSM 事件，state 立即退出
  return result                        // 同时返回给 LLM 的 observation
}
```

#### 场景示例

**ReAct（多轮工具调用）**

```yaml
fsm:
  states:
    - name: react
      type: llm
      tools: [search, calculate]
      # 工具只返回数据，不 emit 事件 → LLM 多轮 loop
      # LLM 输出文字 → DONE → 无 on.DONE → 等待用户
```

**意图路由（单次 LLM call 分流）**

```yaml
fsm:
  states:
    - name: intent_classification
      type: llm
      tools: [classify_intent]    # emit INTENT_BOOKING / INTENT_CANCEL / INTENT_UNCLEAR
      on:
        INTENT_BOOKING: route_booking
        INTENT_CANCEL:  route_cancel
        INTENT_UNCLEAR: clarify

    - name: route_booking
      type: action
      handler: spawnBookingAgent
      on:
        DONE: intent_classification

    - name: route_cancel
      type: action
      handler: spawnCancelAgent
      on:
        DONE: intent_classification

    - name: clarify
      type: llm
      tools: [submit_clarification]  # emit CLARIFICATION_RECEIVED
      on:
        CLARIFICATION_RECEIVED: intent_classification
      # 无 on.DONE → LLM 追问后等待用户
```

**槽位收集（多轮 + 确定性完整性检查）**

```yaml
fsm:
  states:
    - name: collect_slots
      type: llm
      instructions: |
        收集三个订票信息：出发城市、目的城市、出发日期。
        已收集信息在 workingMemory 中。逐步追问缺失槽位。
      tools: [update_slot]
      # update_slot handler 内检查完整性，齐全后 emit BOOKING_READY
      on:
        BOOKING_READY: confirm_booking
      # 无 on.DONE → 缺槽位时 LLM 追问，等待用户

    - name: confirm_booking
      type: action
      handler: processBookingOrder
      on:
        DONE: collect_slots
```

槽位完整性在 tool handler 中确定性检查，不依赖 LLM 判断：

```typescript
registerTool({
  name: 'update_slot',
  inputSchema: {
    field: { type: 'string', enum: ['departure', 'destination', 'date'] },
    value: { type: 'string' }
  },
  handler: async ({ field, value }, ctx) => {
    ctx.workingMemory.set(`slot.${field}`, value)
    const required = ['departure', 'destination', 'date']
    const missing = required.filter(f => !ctx.workingMemory.has(`slot.${f}`))
    if (missing.length === 0) {
      ctx.emit('BOOKING_READY', Object.fromEntries(
        required.map(f => [f, ctx.workingMemory.get(`slot.${f}`)])
      ))
    }
    return { updated: field, missing }
  }
})
```

**原则**：确定性逻辑（完整性检查、值校验）放 handler，不让 LLM 负责"什么时候结束"。

#### 扩展原则

高层 DX 工具（可视化编辑器、preset 编译器、内置实体类型库）在 Runtime 之外实现，通过 config 预处理将高层描述展开为规范的 `type: llm/action` 格式后再交给 Runtime。Runtime 不感知 preset，核心不因 DX 层扩展而修改。

```
开发者写 preset DSL
    ↓ config 预处理（编译期）
规范 FSM 定义（type: llm/action + on: + handler）
    ↓
Runtime 执行
```

---

## 7. State Store（可插拔）

### 接口：薄 KV 语义

State Store 只暴露通用 KV 操作，Checkpoint 的序列化、key 结构、历史查询等逻辑全部由上层（Agent Runtime）处理。

```typescript
interface IStateStore {
  set(key: string, value: unknown, ttl?: number): Promise<void>
  get(key: string): Promise<unknown>
  delete(key: string): Promise<void>
  exists(key: string): Promise<boolean>
}
```

### 内置实现

| 实现 | 用途 |
|------|------|
| `MemoryStore` | 默认，进程内，重启丢失 |
| `RedisStore` | 跨进程、跨 session |
| `SQLiteStore` | 本地持久化，无需外部依赖 |

### Key 结构（Runtime 层约定）

```
agent:{agentId}:checkpoint:latest         # 最近 checkpoint
agent:{agentId}:checkpoint:{checkpointId} # 历史 checkpoint
agent:{agentId}:memory                    # 工作记忆
```

`checkpointId` 必须全局唯一，不使用 timestamp 作为唯一 ID。Runtime 可以额外记录 sequence/timestamp 用于排序。

### Checkpoint 结构（Runtime 层定义）

```typescript
interface AgentCheckpoint {
  checkpointId: string             // 全局唯一：agentRunId + sequence 或 UUID
  sequence:     number             // 单 run 内单调递增，便于排序和调试
  goal:         string             // 本次 run 的不可变意图
  currentTurn?: Input              // 当前 turn 输入；未完成 turn resume 时需要恢复
  fsm: { currentState: string; stateData: unknown }
  context: {
    history:               Message[]   // 对话历史（已压缩或完整）
    workingMemory:         unknown     // 当前 turn 中间状态
    instructionsSnapshot:  string[]    // 已加载 skill 名称 + section 列表（instructions 内容可按 manifest 重建）
    contextEpoch:          number
  }
  pendingEvents: Event[]
  children: {
    taskId:       string
    agentId:      string
    checkpointId?: string
    status:       'running' | 'success' | 'error' | 'interrupted'
  }[]
  resolvedManifest: ResolvedManifest
  meta: {
    agentId:        string
    agentRunId:     string
    parentAgentId?: string
    timestamp:      number
    traceId:        string
    activeSpanId?:  string
  }
}
```

Checkpoint 时机：yield point 处，见第 9 节。

---

## 8. Trajectory Store（可插拔）

### 与 State Store 的区别

| | State Store | Trajectory Store |
|--|-------------|-----------------|
| 用途 | 中断恢复、resume | 分析、对比、A/B test |
| 写入模式 | 可覆盖 | append-only |
| 读取模式 | 按 key 精确读 | 按 agentId/experiment 查询 |
| 生命周期 | 任务级 | 长期保留 |

### 接口：OTel 语义，零强依赖

核心框架不依赖 OTel SDK，自定义等价接口；可选插件包按需引入。

```typescript
interface ITrajectoryRecorder {
  startSpan(name: string, attributes?: SpanAttributes): Span
  endSpan(span: Span, status?: 'ok' | 'error'): void
  recordEvent(span: Span, name: string, attributes?: SpanAttributes): void
}
```

### Span 类型（遵循 OTel GenAI Semantic Conventions）

| Span 类型 | 触发时机 | 属性 |
|-----------|---------|------|
| `agent.run` | Agent 整次运行（root span） | `agentId`, `agentVersion`, `goal`, `traceId`, `contextId` |
| `fsm.transition` | FSM 状态转移 | `fromState`, `toState`, `event` |
| `llm.call` | Model Gateway 调用 | `provider`, `model`, `adapter`, `inputTokens`, `outputTokens`, `cost`, `requestId`, `turn`, `loadedSkills`（string[]，当前 instructions bucket 已加载的 skill 名称）, `cacheBreakpoint2Hash`（instructions bucket 内容哈希，epoch 变化时改变）|
| `tool.call` | Tool 执行 | `toolName`, `toolCallId`（LLM 分配的 tool_use ID）, `input`, `output`, `duration`, `turn`, `parallelBatchId`（同一 allSettled 批次内并发 tool 共享同一 batchId）|
| `agent.spawn` | 启动 sub-agent | `childAgentId`, `taskId`, `childTraceId`（子 Agent trajectory 的 traceId）, `resultStatus`（`success` \| `error` \| `interrupted`）, `turn` |

### Versioning 模型

**组件各自独立版本（semver），AgentConfig 精确 pin 所有依赖版本。**

改变任何依赖版本 = 产生新的 AgentConfig 版本。组件版本的价值不是绕过 agent 版本，而是让版本间的 diff 精确可追溯。

```yaml
# AgentConfig（agent.md frontmatter）
version: "1.2.0"
fsm:
  states:
    - name: react
      type: llm
toolboxes:
  browser:  "2.0.0"
skills:
  research: "1.1.0"     # 从 1.0.0 升级
  summarize: "1.0.0"
sub_agents:
  researcher: "1.0.0"
  coder:      "2.1.0"
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

版本 diff 示例（v1.1.0 → v1.2.0）：
```
skill:research      1.0.0 → 1.1.0   ← 唯一变化，行为差异归因于此
toolbox:browser     2.0.0 → 2.0.0
subAgent:researcher 1.0.0 → 1.0.0
model               unchanged
system_prompt       unchanged
```

### Trajectory 结构

```typescript
interface Trajectory {
  traceId:      string
  agentId:      string
  // 运行时 resolved manifest —— 完整依赖快照，确保可复现
  resolvedManifest: ResolvedManifest
  startTime:    number
  endTime?:     number
  status:       'running' | 'completed' | 'interrupted' | 'failed'
  spans:        Span[]          // append-only
  metrics?: {
    totalTokens: number
    totalSteps:  number
    duration:    number
    cost:        number
  }
}
```

```typescript
interface ResolvedDependency {
  version: string
  source:  'local' | 'workspace' | 'registry'
  sha:     string        // 内容哈希或 registry integrity
}

interface ResolvedManifest {
  agentVersion:    string
  systemPromptSha: string
  model: {
    provider: string
    model:    string
    adapter:  ResolvedDependency
    optionsSha?: string
  }
  skills:    Record<string, ResolvedDependency>
  toolboxes: Record<string, ResolvedDependency>
  subAgents: Record<string, ResolvedDependency>   // agentId → resolved version + sha
}
```

Checkpoint 引用 `traceId` + 当前 span ID，建立恢复点与执行历史的对应关系。

### 内置实现

| 实现 | 用途 |
|------|------|
| `NoopRecorder` | 默认，零开销 |
| `JSONLRecorder` | 写本地 JSONL 文件，无外部依赖 |
| `ConsoleRecorder` | 开发调试用 |

### 可选插件（独立包，按需引入）

```
@milkie/recorder-otel        → 接入 OTel SDK / Collector
@milkie/recorder-langsmith   → 接入 LangSmith
@milkie/recorder-braintrust  → 接入 Braintrust
```

### A/B Test / Experiment 模型

```typescript
interface Experiment {
  id:       string
  goal:     string              // 控制变量：所有 variant 使用相同的 goal（来自 AgentInvokeRequest.goal）
  variants: {
    name:         string
    agentVersion: string        // 不同的 AgentConfig 版本
    trajectoryIds: string[]
  }[]
}
```

同一个 goal 用不同 AgentConfig 版本跑，产生不同 trajectories。因为依赖树精确记录，对比时可明确归因：行为差异来自哪个组件的哪次版本变化。

---

## 9. Interrupt 模型

### 核心原则

- **Interrupt ≠ Error**：两者 FSM 转移路径完全分开
- 用户中断采用**立即打断**模式（类 Claude Code Escape 键）
- 中断在**协作式 yield point** 处理，不强制抢占 async 操作

### 协作式事件检查（Yield Points）

执行流在以下位置主动检查事件队列：

```
yield points：
  ├─ tool 执行前
  ├─ tool 执行后
  ├─ LLM call 前
  ├─ LLM response 处理前
  └─ FSM 状态转移时
```

每个 yield point 执行相同逻辑：

```typescript
async function checkEvents(): Promise<void> {
  const event = eventQueue.dequeue()
  if (!event) return

  if (event.type === 'interrupt') {
    await saveCheckpoint()        // 状态已知安全，持久化
    fsm.emit('interrupt', event)  // FSM 转移到 paused
    throw new InterruptSignal()   // 中断当前执行流
  }

  // 用户消息、Workflow 回调等其他事件放入待处理队列
  pendingEvents.push(event)
}
```

**关键特性**：中断只发生在已知安全状态（tool 完整执行后），无需 AbortController 强制取消。Checkpoint 和事件检查天然发生在同一位置。

### FSM 全局转移规则

```
ANY_STATE + error      → error_handling   （错误恢复，可重试）
ANY_STATE + interrupt  → paused           （干净停止 + checkpoint）
```

`error_handling` 和 `paused` 是框架自动注入的保留状态，无需在 `fsm.states` 中声明。`error_handling` 默认行为：重试触发 error 的 tool call（最多 3 次，仅 `retryable: true` 的错误），超限后转移到 `failed`（terminal）。

### 中断传播（Supervisor Tree）

- **向下**：父 Agent 中断 → 将 interrupt 事件写入所有子 Agent 的 eventQueue → 子 Agent 在各自下一个 yield point 处理
- **向上**：子 Agent 中断 → Task tool 返回 `{ status: 'interrupted', checkpointId }` → 父 FSM 在下一个 yield point 收到并处理

### Event Queue

所有外部信号统一进入 Agent 的 event queue，由 yield point 处理：

| 事件类型 | 来源 | 处理 |
|---------|------|------|
| `interrupt` | 用户、父 Agent | 立即 checkpoint + paused |
| `user_message` | 用户输入 | 追加到待处理队列，下一 turn 处理 |
| `workflow_callback` | Workflow System | 追加到待处理队列 |
| `resume` | 用户或系统 | 从 checkpoint 恢复，继续执行 |

---

## 10. Workflow Bridge

> **状态：计划中（v2）。** 当前版本不实现此接口；Agent 升级人工等场景直接由 `escalated` terminal state 输出说明文本。

与外部 Workflow System 双向嵌入的接口层，两个系统各自独立，通过协议通信，不共享状态。

### Agent → Workflow

Agent 通过系统内置 `WorkflowBridge` tool 触发业务流程：

```typescript
// WorkflowBridge tool 的输入参数（LLM 只填前四个字段）
interface WorkflowTriggerInput {
  workflowId: string
  trigger:    string    // 触发事件名，Workflow 侧定义
  payload:    unknown   // 任意 JSON，业务数据
  mode:       'fire_and_forget' | 'await_completion'
  // 以下字段由 Framework 自动注入，LLM 不感知
  agentContextId: string   // 当前 Agent 的 contextId；Workflow 回调时传入可复用同一 context
  agentGoal:      string   // 当前 Agent 的 goal；Workflow 回调时传入可保持 goal 不变性
}

// tool 返回结果
interface WorkflowTriggerResult {
  runId:   string       // Workflow 运行实例 ID
  result?: unknown      // 仅 mode='await_completion' 时有值
}
```

### Workflow → Agent

milkie 暴露同步调用接口，Workflow 作为调用方：

```typescript
// milkie 对外暴露的调用接口（HTTP / SDK）
interface AgentInvokeRequest {
  agentId:    string    // 目标 Agent 配置 ID
  goal:       string    // 不可变的运行意图，写入 agent.run span，供 Experiment 比较
  input:      Input     // 当前 turn 的动态输入；多轮时可变化
  contextId?: string    // 可选：复用已有 context（多轮场景）
}

interface AgentInvokeResponse {
  output:     string    // Agent 最终输出（text）
  agentRunId: string    // 本次运行 ID，可关联 Trajectory
  contextId:  string    // 本次 run 所属 context；多轮场景传入下次 AgentInvokeRequest.contextId 复用历史
}
```

`goal` 是 run 级控制变量，创建 run 后不可变；`input` 是 turn 级输入，可随多轮交互变化。复用 `contextId` 时，如果不显式创建新 run，则沿用原 goal，并把新的 `input` 写入 `current_turn`。

WorkflowBridge 是薄适配器，负责协议转换；不持有状态，不影响 Agent 身份。

---

## 11. 完整架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent Interface                           │  Public API
├──────────────────────────────────────────────────────────────┤
│                    Agent Runtime                             │  事件循环 · 并发 · 生命周期
├──────────────────────┬───────────────────────────────────────┤
│     FSM Layer        │   LLM Context Layer                  │  核心两层（正交）
│                      │  ┌─────────────────────────────────┐ │
│  当前状态             │  │ system_prompt   [cached ①]      │ │
│  转移规则             │  │ instructions    [cached ②]      │ │
│  硬/软/事件转移       │  │ history                         │ │
│                      │  │ working_memory                  │ │
│                      │  │ current_turn                    │ │
│                      │  └─────────────────────────────────┘ │
├──────────────────────┴───────────────────────────────────────┤
│                    Model Gateway                             │  Provider adapter / stream 归一化
├──────────────────────────────────────────────────────────────┤
│                    Tool Registry                             │  系统 / Toolbox / Cognitive
├──────────────────────────────────────────────────────────────┤
│               Skill Registry（Resource 层）                  │  Skill 定义 + Agent Templates
├──────────────────────┬───────────────────────────────────────┤
│   State Store        │   Trajectory Store                   │  可插拔，相互独立
│   中断恢复 / resume  │   分析 / A-B test / 溯源              │
│   Memory/Redis/DB    │   JSONL / OTel / LangSmith           │
└──────────────────────┴───────────────────────────────────────┘
                              ↕
                    Workflow Bridge（外部）
```

---

## 12. 部署模型

### 执行模型

Runtime 默认运行在单 Node.js 进程，所有并发基于 async/await（coroutine 语义）。LLM API call 和 Tool 执行均为 I/O 密集型，事件循环可高效处理大量并发 Agent run。

CPU 密集型 Tool（本地模型推理、大文件处理）会阻塞事件循环，v2 将通过 `ToolDefinition.executionMode` 路由到 Worker thread 或子进程；v1 暂不实现。

### 水平扩展

**Redis State Store 是无状态计算的关键**：Agent 运行时状态（FSM、Context、Checkpoint）全部外置在 Redis，计算节点本身无状态。

```
Client
  ↓
HTTP Load Balancer / Work Queue
  ↓              ↓              ↓
Worker-1        Worker-2       Worker-N     ← 无状态，可任意扩缩
(Node.js)       (Node.js)      (Node.js)
         ↓
  Shared Redis State Store
  Shared Trajectory Store
```

任意 Worker 都可以处理任意 `AgentInvokeRequest`，也可以 resume 任意 Checkpoint——不依赖本地状态。

### Sub-agent Dispatch 策略

`AgentConfig` 预留 `dispatch` 字段，控制 sub-agent 的执行位置：

```typescript
interface AgentConfig {
  // ...
  dispatch?: 'local'   // 默认：同进程 spawn（低延迟，开发 / 小规模）
           | 'queue'   // 投递到 Work Queue，由远端 Worker 执行（水平扩展）
}
```

`dispatch: queue` 的具体实现（Bull / SQS / 自研）及 sub-agent 跨进程结果回传机制见独立 Spec。

### 模式对比

| 模式 | State Store | Sub-agent dispatch | 适用场景 |
|------|------------|-------------------|---------|
| 单进程 | Memory / SQLite | local | 开发、单机小规模 |
| 多 Worker | Redis | local | 中等规模，同机器多进程 |
| 分布式 | Redis | queue | 大规模，跨机器 |

---

## 13. E2E 验证场景

E2E 场景独立维护，见 [agent-e2e-scenarios.md](./2026-05-16-agent-e2e-scenarios.md)。

包含 6 个场景及完整能力覆盖矩阵。

---

## 待定设计（独立 Spec）

- `interrupt-lifecycle-design.md`：中断传播、AbortController 时序、并发 tool join 细节、Supervisor Tree 边界情况
- `dispatch-queue-design.md`：Sub-agent 跨进程 dispatch、Work Queue 集成、结果回传机制
