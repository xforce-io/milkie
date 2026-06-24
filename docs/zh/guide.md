# milkie 使用指南

本指南描述 milkie **当前实现**的用法。完整 target 架构——包括
run-as-product 立场、event-sourced Agent Trace、IOPort、Evolution、
以及 cross-cutting invariants——见
[ARCHITECTURE.md](../../ARCHITECTURE.md)。

- [1. 核心概念](#1-核心概念)
- [2. Agent 配置](#2-agent-配置)
- [3. State Store](#3-state-store)
- [4. 模型适配器](#4-模型适配器)
- [5. 内置工具](#5-内置工具)
- [6. 多 Agent 模式](#6-多-agent-模式)
- [7. 中断与恢复](#7-中断与恢复)
- [8. Trajectory 与可观测性](#8-trajectory-与可观测性)
  - [概念:可观测 vs 血缘](./concepts-observability-vs-lineage.md) — 「这结论哪来的、能信吗」用哪个回答
- [9. API 参考](#9-api-参考)

---

## 1. 核心概念

### Agent = FSM

milkie 中的每个 Agent 都是一个有限状态机。FSM 只有两种状态类型：

**`type: llm`** — 循环调用 LLM，直到以下情况之一：
- 工具调用 `ctx.emit(event)` → FSM 按 `on` 中声明的规则转移到目标状态
- LLM 输出纯文本 → FSM 内部触发 `DONE`；如果 `on.DONE` 有目标则转移，否则等待下一条用户消息（多轮对话模式）
- 达到 `max_iterations` → 抛出 `MaxIterationsError`，Agent run 返回 `status: 'error'`

**`type: action`** — 执行确定性 handler，不调用 LLM，执行完成后按 `DONE` 或 `ERROR` 转移。

```
type: llm 状态                      type: action 状态
──────────────────                  ─────────────────
  LLM 调用                             handler()
     ↓                                    ↓
  tool_use? → 执行工具               触发 DONE / ERROR
     ↓                                    ↓
  ctx.emit(event)?               on.DONE → 下一状态
     ↓ 是 → 状态转移
  纯文本? → 触发 DONE
     ↓
  on.DONE → 下一状态（或等待）
```

### FSM 保留状态

以下两个保留状态自动可用，无需手动声明：

| 保留状态 | 触发条件 | 行为 |
|----------|---------|------|
| `error_handling` | 工具抛出带 `retryable: true` 的错误 | 运行时临时转入此状态，等待 500 ms 后对同一工具重试（最多 3 次）。不可重试的错误跳过 `error_handling`，以 `isError: true` 的 observation 形式反馈给 LLM。 |
| `paused` | 调用 `milkie.interrupt(contextId)` | 保存 checkpoint，干净停止执行，`AgentResult.status` 为 `'interrupted'`。 |

### 并行执行

**Intra-agent 并行**（单 Agent 内）：LLM 在单次响应中输出多个 `tool_use` block 时，所有标记了 `parallelSafe: true` 的工具并发执行。未标记的工具串行执行。所有工具以 `allSettled` 语义 join——某个工具失败不会取消其他工具。

**Inter-agent 并行**（跨 Agent）：Orchestrator Agent 在单次 LLM 响应中调用多个子 Agent 工具时，框架并发启动各子 Agent 实例，每个实例拥有独立的 FSM 和 Context，内部状态完全隔离。

### Context 分层结构

> **概念模型。** 五个 bucket 的排列顺序反映了 `ContextLayer` 构造 LLM 请求的方式。Provider 级别的 prefix cache 和历史压缩策略在 v1 中尚未实现。

每个 Agent 的 LLM Context 按以下顺序组织为五个 bucket：

```
[稳定区]
  system_prompt       在 Agent 生命周期内从不改变
  instructions        已加载 Skill 的指令（epoch 边界生效）

[动态区 — 每 turn 重建]
  history             累积的对话历史
  working_memory      当前 turn 中间状态
  current_turn        当前用户输入（永远在末尾）
```

### Goal 与 Input 的区别

- **`goal`** — Agent run 的不可变意图。写入 `agent.run` span，用于 A/B 实验跨版本对比。多轮对话中不会改变。
- **`input`** — 当前 turn 的动态输入。每次用同一 `contextId` 调用 `invoke()` 时可以变化。

---

## 2. Agent 配置

### TypeScript 对象

```typescript
import type { AgentConfig } from 'milkie'

const config: AgentConfig = {
  agentId:      'my-agent',
  version:      '1.0.0',
  systemPrompt: '你是一个乐于助人的助手。',
  fsm: {
    states: [
      {
        name:           'react',
        type:           'llm',
        max_iterations: 15,
        // tools: ['web_search']  // 省略则继承 Agent 的全部工具
      },
    ],
  },
  model: {
    provider: 'volcengine',
    model:    'doubao-seed-2.0-lite',
    adapter:  'openai-compatible',
    baseUrl:  process.env['VOLCENGINE_API_BASE'],
  },
  subAgents: {
    'researcher': '1.0.0',   // agentId → 固定版本
  },
}
```

### Markdown Frontmatter 文件

Agent 也可以定义为带 YAML frontmatter 的 `.md` 文件，文件正文作为 `systemPrompt`。

```markdown
---
agentId: my-agent
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 15
model:
  provider: volcengine
  model: doubao-seed-2.0-lite
  adapter: openai-compatible
  baseUrl: https://your-endpoint
sub_agents:
  researcher: "1.0.0"
---

你是一个乐于助人的助手。
```

用以下方式加载：

```typescript
const config = milkie.loadAgentFile('./agents/my-agent.md')
```

### FSMState 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | `string` | 是 | FSM 内唯一的状态名 |
| `type` | `'llm' \| 'action'` | 是 | 执行类型 |
| `instructions` | `string` | 否 | 仅在该状态中注入的额外指令（type: llm） |
| `tools` | `string[]` | 否 | 该状态可用的工具名列表；省略则继承 Agent 所有工具 |
| `on` | `Record<string, string>` | 否 | 事件 → 目标状态的转移映射 |
| `handler` | `string` | 否 | type: action 时执行的 handler 名称（通常为子 Agent 的 agentId） |
| `terminal` | `boolean` | 否 | 为 true 时该状态为终止状态，无出口 |
| `max_iterations` | `number` | 否 | LLM loop 最大迭代次数（type: llm），默认不限制 |

---

## 3. State Store

State Store 用于持久化 checkpoint，支持中断恢复和多轮对话历史。所有 Store 均实现 `IStateStore` 接口。

### MemoryStore（默认）

进程内存储，无外部依赖，进程重启后状态丢失。

```typescript
import { Milkie, MemoryStore } from 'milkie'

const milkie = new Milkie({ stateStore: new MemoryStore() })
```

### SQLiteStore

本地持久化，无需外部服务。

```typescript
import { Milkie, SQLiteStore } from 'milkie'

const store = new SQLiteStore({ path: './data/state.db' })
await store.init()   // 首次运行时自动建表

const milkie = new Milkie({ stateStore: store })
// 用完后关闭
store.close()
```

### RedisStore

跨进程、跨 session。水平扩展部署时必选。

```typescript
import { Milkie, RedisStore } from 'milkie'

const store = new RedisStore({
  host: 'localhost',
  port: 6379,
  // password: '...',
  // db: 0,
})
await store.init()   // 使用前必须调用

const milkie = new Milkie({ stateStore: store })
// 用完后断开
await store.disconnect()
```

### 自定义 Store

实现 `IStateStore` 接口即可接入自己的存储后端：

```typescript
interface IStateStore {
  set(key: string, value: unknown, ttl?: number): Promise<void>
  get(key: string): Promise<unknown>
  delete(key: string): Promise<void>
  exists(key: string): Promise<boolean>
}
```

---

## 4. 模型适配器

### Anthropic

```typescript
model: {
  provider: 'anthropic',
  model:    'claude-sonnet-4-6',
  adapter:  'anthropic',
  // baseUrl: 'https://api.anthropic.com',  // 可选
}
```

需要在环境变量中设置 `ANTHROPIC_API_KEY`。

### OpenAI-Compatible

支持所有 OpenAI 兼容接口：OpenAI、Azure OpenAI、火山引擎（豆包）、DeepSeek、本地 Ollama 等。

```typescript
// OpenAI
model: {
  provider: 'openai',
  model:    'gpt-4o',
  adapter:  'openai-compatible',
  baseUrl:  'https://api.openai.com/v1',
}
```

```typescript
// 火山引擎（豆包）
model: {
  provider: 'volcengine',
  model:    'doubao-seed-2.0-lite',
  adapter:  'openai-compatible',
  baseUrl:  process.env['VOLCENGINE_API_BASE'],
}
```

适配器按以下顺序读取 API Key：`VOLCENGINE_TOKEN`，然后是 `OPENAI_API_KEY`。运行前设置对应的环境变量：

```bash
export OPENAI_API_KEY=sk-...
# 或者
export VOLCENGINE_TOKEN=your-token
export VOLCENGINE_API_BASE=https://your-endpoint/v1
```

### 自定义 Gateway

实现 `IModelGateway` 接口，通过 `gateway` 选项注入后覆盖所有 Agent 的模型调用。适合测试或接入自定义模型服务。

```typescript
import type { IModelGateway, ModelRequest, ModelResponse } from 'milkie'

class MockGateway implements IModelGateway {
  async complete(req: ModelRequest): Promise<ModelResponse> {
    return {
      content:      [{ type: 'text', text: 'mock 响应' }],
      toolCalls:    [],
      finishReason: 'end_turn',
    }
  }

  async *stream(req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

const milkie = new Milkie({ gateway: new MockGateway() })
```

---

## 5. 内置工具

### Cognitive Toolbox

`cognitiveTools` 和 `systemTools` 由运行时**自动注册**，每个 Agent 开箱即用，无需手动传入 `Milkie`。

| 工具 | 说明 |
|------|------|
| `think` | 将推理步骤记录到工作记忆，无副作用，`parallelSafe: true` |
| `create_plan` | 创建任务步骤清单，存储在工作记忆中，多步骤任务开始时调用一次 |
| `update_step` | 将某步骤标记为 `done` 或 `failed`；失败时重新调用 `create_plan` 修订计划 |

**System prompt 中的使用说明建议：**

```
遇到多步骤任务时：
1. 首先调用 create_plan 列出所有步骤。
2. 依次执行，每步完成后调用 update_step 标记。
3. 所有步骤完成后输出最终结果。
```

### System Tools

同样自动注册，无需手动配置。

| 工具 | 说明 |
|------|------|
| `skill_list` | 返回可用 Skill 列表。（v1 stub — 当前始终返回空列表，完整 Registry 支持计划在 v2 实现。） |
| `skill_request` | 请求在下一个 context epoch 加载某个 Skill。需要在 `AgentConfig.skills` 中声明该 Skill，并通过 `AgentConfig.skillInstructions` 提供指令内容。指令从下一 turn 起生效。 |

在 Agent 配置中声明 Skill：

```typescript
const config: AgentConfig = {
  // ...
  skills: { research: '1.1.0' },
  skillInstructions: {
    research: `## 研究指南
搜索相关信息，以带引用的格式总结发现。`,
  },
}
```

### 自定义工具

```typescript
import type { ToolDefinition } from 'milkie'

const myTool: ToolDefinition = {
  name:        'query_database',
  description: '根据 ID 从数据库查询记录。',
  inputSchema: {
    type:       'object',
    properties: {
      id: { type: 'string', description: '记录 ID' },
    },
    required: ['id'],
  },
  parallelSafe: true,   // true 时允许与其他工具并发执行
  handler: async (input, ctx) => {
    const { id } = input as { id: string }

    // ctx.workingMemory — 读写当前 turn 的中间状态
    // ctx.emit(event)   — 触发 FSM 状态转移
    // ctx.stateStore    — 读写持久化状态

    const record = await fetchRecord(id)
    ctx.workingMemory.set('lastQueried', id)

    if (!record) {
      ctx.emit('NOT_FOUND')
      return { found: false }
    }

    return { found: true, record }
  },
}
```

**`ToolContext` 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `workingMemory` | `WorkingMemory` | 当前 turn 的暂存区，随 checkpoint 持久化 |
| `emit` | `(event, payload?) => void` | 触发 FSM 状态转移 |
| `stateStore` | `IStateStore` | 持久化键值存储 |
| `agentFactory` | `AgentFactory` | 以编程方式启动子 Agent |

---

## 6. 多 Agent 模式

### 声明子 Agent

在 `AgentConfig.subAgents` 中声明子 Agent，框架自动为每个子 Agent 生成一个同名工具，Orchestrator 的 LLM 像调用普通工具一样调用它们。

```typescript
const orchestratorConfig: AgentConfig = {
  agentId:      'orchestrator',
  version:      '1.0.0',
  systemPrompt: `你负责协调研究任务。
在一次响应中同时调用 researcher 和 writer，再汇总它们的输出。`,
  fsm: {
    states: [{ name: 'react', type: 'llm', max_iterations: 10 }],
  },
  model: { /* ... */ },
  subAgents: {
    'researcher': '1.0.0',
    'writer':     '1.0.0',
  },
}

milkie.registerAgent(researcherConfig)
milkie.registerAgent(writerConfig)
milkie.registerAgent(orchestratorConfig)
```

当 Orchestrator 的 LLM 在单次响应中同时调用 `researcher` 和 `writer` 时，框架并发启动它们，各自运行于独立的 FSM + Context 实例中。

### 并发调用提示

在 System prompt 中明确指导 LLM 并发调用：

```
在一次响应中同时调用 researcher 和 writer，不要分开调用。
```

框架自动处理并发，所有子 Agent 完成后（`allSettled` 语义）统一将结果作为 observation 返回给 Orchestrator。

### 子 Agent 工具 Schema

自动生成的子 Agent 工具接受以下参数：

```typescript
{
  goal:  string   // 子 Agent 本次 run 的不可变意图
  input: string   // 当前 turn 的动态输入
}
```

### 用 Action State 做确定性路由

无需 LLM 时，用 `type: action` 直接路由到子 Agent：

```typescript
{
  name:    'route_to_billing',
  type:    'action',
  handler: 'billing-specialist',   // 对应已注册的 agentId
  on: { DONE: 'completed' },
}
```

### Context 隔离原则

子 Agent 无法读写父 Agent 的 working memory 或对话历史，信息通过显式传递：
- **父 → 子**：通过 `goal` 和 `input` 参数传入
- **子 → 父**：通过工具返回值注入父 Agent 的 context

---

## 7. 中断与恢复

### 中断运行中的 Agent

```typescript
const contextId = `ctx-${Date.now()}`

// 启动长任务
const runPromise = milkie.invoke({
  agentId: 'analyst',
  goal:    '处理 1000 条记录',
  input:   '开始处理',
  contextId,
})

// 任意时刻中断
setTimeout(() => milkie.interrupt(contextId), 5000)

const result = await runPromise
// result.status === 'interrupted'
// checkpoint 保存在：context:{contextId}:checkpoint:latest
```

中断是协作式的：运行时在 yield point（每次工具调用前后、每次 LLM 调用前）检查中断信号。正在执行的工具调用会完整跑完，下一个 yield point 处才真正停止并保存 checkpoint。

checkpoint key 格式：
```
context:{contextId}:checkpoint:latest
```

### 恢复执行

```typescript
const checkpointKey = `context:${contextId}:checkpoint:latest`

const result = await milkie.resume(
  checkpointKey,
  'analyst',
  '处理 1000 条记录',    // 与原 goal 保持一致
  '继续处理剩余记录',
)
// result.status === 'completed'
```

恢复后的 run 使用相同的 `contextId` 和 `agentRunId`，Trajectory span 续写到同一条 trace 中。

### 多轮对话（无中断）

跨多次 `invoke()` 复用同一 `contextId`，对话历史自动保留：

```typescript
// 第一轮
const turn1 = await milkie.invoke({
  agentId: 'assistant',
  goal:    '协助用户处理订单 #12345',
  input:   '我的订单好像延误了。',
})

// 第二轮 — 复用 contextId，历史自动延续
const turn2 = await milkie.invoke({
  agentId:   'assistant',
  goal:      '协助用户处理订单 #12345',
  input:     '能帮我查一下物流状态吗？',
  contextId: turn1.contextId,
})
```

---

## 8. Trajectory 与可观测性

### 开启 Span 记录

```typescript
import { Milkie, TrajectoryStore } from 'milkie'

const trajectoryStore = new TrajectoryStore({
  jsonlDir: './trajectories',   // 每次 run 写一个 JSONL 文件
})

const milkie = new Milkie({ trajectoryStore })
```

### Span 类型

| Span | 触发时机 | 属性 |
|------|---------|------|
| `agent.run` | 整次 Agent 执行（根 span） | `agentId`, `goal`, `contextId` |
| `fsm.transition` | 每次 FSM 状态转移 | `fromState`, `toState`, `event` |
| `llm.call` | 每次模型 API 调用 | `provider`, `model`, `turn`, `state`, `loadedSkills`, `contextEpoch`；token 用量以 `usage` event 记录（非 span attribute） |
| `tool.call` | 每次工具执行 | `toolName`, `toolCallId`, `input`, `turn`, `attempt`, `parallelBatchId`；成功时追加 `output`；duration = `span.endTime - span.startTime` |
| `agent.spawn` | 每次子 Agent 启动 | `childAgentId`, `taskId`, `childTraceId`, `childContextId`, `turn`；完成后追加 `resultStatus` 和 `checkpointId` |

### 查询 Trajectory

```typescript
// 按 runId 查询（最精确）
const traj = await trajectoryStore.getByRunId(result.agentRunId)

// 按 contextId 查询（获取一次对话的所有 run）
const traj = await trajectoryStore.getByContextId(contextId)

// 筛选特定类型的 span
const toolSpans = traj.spans.filter(s => s.name === 'tool.call')
const llmCalls  = traj.spans.filter(s => s.name === 'llm.call')
```

### ResolvedManifest

每条 Trajectory 在运行时捕获完整的依赖快照：

```typescript
traj.resolvedManifest.agentVersion   // '1.2.0'
traj.resolvedManifest.model.model    // 'doubao-seed-2.0-lite'
traj.resolvedManifest.skills         // { research: { version: '1.1.0' } }
traj.resolvedManifest.subAgents      // { researcher: { version: '1.0.0' } }
```

### A/B 实验

用相同的 goal 跑两个不同版本的 Agent，对比 Trajectory：

```typescript
const [r1, r2] = await Promise.all([
  milkie.invoke({ agentId: 'agent-v1', goal, input: goal }),
  milkie.invoke({ agentId: 'agent-v2', goal, input: goal }),
])

const [t1, t2] = await Promise.all([
  trajectoryStore.getByRunId(r1.agentRunId),
  trajectoryStore.getByRunId(r2.agentRunId),
])

// 对比输出、token 用量、工具调用次数等
console.log(r1.output, r2.output)
console.log(t1.resolvedManifest.skills, t2.resolvedManifest.skills)
```

### 可观测之外:血缘

Trajectory 回答「**发生了什么**」——它跑了哪些工具、输出是什么、可回放可对比。但它回答不了「**报告里这条结论凭哪条源、能不能信**」:那需要**血缘**(claim 带一条防伪造的边指向它的源)。两者建在同一份 trace 上,职责不同,极易混淆。

→ 概念与取舍详见 **[可观测 vs 血缘](./concepts-observability-vs-lineage.md)**(含一个易踩的陷阱:cite 到「整坨输出」算血缘吗?)。

---

## 9. API 参考

### `new Milkie(options?)`

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stateStore` | `IStateStore` | `new MemoryStore()` | 持久化 checkpoint 和多轮历史 |
| `gateway` | `IModelGateway` | `undefined` | 覆盖所有 Agent 的 gateway（适合测试） |
| `tools` | `ToolDefinition[]` | `[]` | 所有已注册 Agent 可用的工具 |
| `trajectoryStore` | `TrajectoryStore` | `undefined` | 记录 span；不传则禁用追踪 |

### `milkie.registerAgent(config)`

注册 `AgentConfig`，必须在 `invoke()` 之前调用。已注册的 Agent 自动作为其他 Orchestrator 的子 Agent 可用。

### `milkie.loadAgentFile(filePath)`

从带 YAML frontmatter 的 Markdown 文件加载 Agent，返回解析后的 `AgentConfig` 并自动注册。

### `milkie.invoke(request)`

```typescript
interface AgentInvokeRequest {
  agentId:    string    // 已注册的 Agent ID
  goal:       string    // 不可变的 run 意图
  input:      string    // 当前 turn 的输入
  contextId?: string    // 省略则开启新对话
}

interface AgentResult {
  agentRunId: string
  contextId:  string
  output:     string
  status:     'completed' | 'interrupted' | 'error'
}
// status 为 'interrupted' 时，checkpoint 保存在：
// context:{result.contextId}:checkpoint:latest
```

### `milkie.resume(checkpointKey, agentId, goal, input)`

从保存的 checkpoint 恢复中断的 run，继续使用相同的 `agentRunId` 和 `contextId`。

```typescript
const result = await milkie.resume(
  'context:ctx-abc:checkpoint:latest',
  'my-agent',
  '原始 goal',
  '继续处理',
)
```

### `milkie.interrupt(contextId)`

向绑定了 `contextId` 的 Agent 发送中断信号，同时传播到所有正在运行的子 Agent。立即返回，实际停止发生在下一个 yield point 处。

### `milkie.registerTool(tool)`

在构造后向运行时追加单个工具，等价于在 `options.tools` 中传入。

### `milkie.getAgent(agentId)`

返回指定 ID 的 `AgentConfig`，不存在则返回 `undefined`。
