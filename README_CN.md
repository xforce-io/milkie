# milkie

**TypeScript LLM Agent 库 — 每个 run 都是一等工程产品。**

milkie 是一个 TypeScript 库，用于构建 LLM 驱动的 Agent。**核心承诺：每个
Agent run（完整推理轨迹，不只是最终 output）是一等工程产品**——可寻址、
可复现、可分叉、可对比、可归因。

底层每个 Agent 模式（对话、ReAct、多状态工作流、多 Agent 编排）都是同一
个有限状态机（FSM）在同一个 thin runtime 上跑。Runtime、event-sourced 的
**Agent Trace**、确定性 **Evolution** 实验子系统共同构成三个 peer 子系统。

[English](./README.md) · [使用指南](./docs/zh/guide.md) · [架构文档](./ARCHITECTURE.md)

---

## 核心特性（当前已实现）

- **Agent = FSM** — 只有两种 state 类型（`llm` / `action`），组合出所有 Agent 模式，无需特殊分支
- **Intra-agent 并行** — LLM 单次响应输出多个 `tool_use`，运行时并发执行，`allSettled` join 后统一返回
- **Inter-agent 并行** — 子 Agent 以具名 tool 形式暴露给 Orchestrator，跨独立 FSM + Context 并发启动
- **中断与恢复** — 协作式 yield point 自动存档，任意中断的运行都能从原点续跑
- **多轮对话** — 跨 `invoke()` 复用同一 `contextId`，历史自动保留
- **Event-sourced Agent Trace** — append-only event log 记录每次 LLM / tool I/O，带 `causedBy` 因果链；`Milkie.replay(runId)` 从 log 重跑已记录的 run，**零真实 LLM 调用**（Phase 3 结构等价 replay）
- **`milkie` CLI** — `agent list / run / resume / interrupt` + `trace inspect / replay`，基于 `.milkie/agents.json` manifest；默认持久化 SQLite state store，interrupt / resume 跨 CLI 进程也能跑。CLI 是 agent 消费者的 canonical surface（ARCHITECTURE.md invariants 12–13），每个 verb 跟 SDK 调用 1:1 对应
- **可插拔后端** — 按需切换 State Store（Memory / SQLite / Redis）和 Trajectory Recorder（JSONL / 内存 / 控制台）
- **多 Provider** — 开箱支持 Anthropic 和所有 OpenAI-compatible 接口

**仍在开发的 Target 能力** —— fork / diff / lineage 作为 event log 上的
一等操作、用于 byte-identical replay 的 non-determinism log、以及
Evolution 实验子系统。当前实现 vs. target 架构的对账详见
[ARCHITECTURE.md → Implementation Status](./ARCHITECTURE.md#implementation-status)。

---

## 安装

```bash
npm install milkie
```

需要 Node.js ≥ 20。

---

## 快速开始

```typescript
import { Milkie, MemoryStore } from 'milkie'
import type { AgentConfig, ToolDefinition } from 'milkie'

// 1. 定义工具
const webSearch: ToolDefinition = {
  name: 'web_search',
  description: '搜索网络获取信息。',
  inputSchema: {
    type: 'object',
    properties: { query: { type: 'string' } },
    required: ['query'],
  },
  parallelSafe: true,
  handler: async (input) => {
    const { query } = input as { query: string }
    return { result: `"${query}" 的搜索结果` }  // 替换为真实搜索
  },
}

// 2. 配置 Agent
const researchAgent: AgentConfig = {
  agentId: 'researcher',
  version: '1.0.0',
  systemPrompt: '你是一个研究助手，使用 web_search 准确回答问题。',
  fsm: {
    states: [{ name: 'react', type: 'llm', max_iterations: 10 }],
  },
  model: {
    provider: 'volcengine',
    model: 'doubao-seed-2.0-lite',
    adapter: 'openai-compatible',
    baseUrl: process.env['VOLCENGINE_API_BASE'],
  },
}

// 3. 运行
const milkie = new Milkie({
  stateStore: new MemoryStore(),
  tools: [webSearch],
})
milkie.registerAgent(researchAgent)

const result = await milkie.invoke({
  agentId: 'researcher',
  goal: '总结 TypeScript 5.0 的关键新特性',
  input: 'TypeScript 5.0 有哪些主要新特性？',
})

console.log(result.output)
// result.status: 'completed' | 'interrupted' | 'error'
```

---

## CLI

包装好之后，同一个 agent 也可以直接从 shell 跑。把 manifest 放在
`.milkie/agents.json` 里，从项目内任意目录调 CLI：

```bash
$ cat .milkie/agents.json
{ "agents": [{ "id": "researcher", "file": "../agents/researcher.md" }] }

# 1. 列已注册 agents（启动时 auto-load manifest）
$ milkie agent list
{"id":"researcher","source":"manifest"}

# 2. 执行 agent —— 记录到 .milkie/runs/<runId>.jsonl
$ milkie agent run researcher --input "TypeScript 5.0 有哪些新特性？"
{"runId":"...","contextId":"...","status":"completed","lastOutput":"..."}

# 3. Replay 已记录的 run —— 零真实 LLM 调用
$ milkie trace replay <runId>
{"newRunId":"...","status":"completed","output":"..."}

# 4. 按 JSONL 输出 run 里的每一个 event
$ milkie trace inspect <runId>
{"id":"...","runId":"...","type":"agent.run.started",...}
{"id":"...","runId":"...","type":"llm.requested",...}
...
```

可用 verbs：`agent list / run / resume / interrupt`、
`trace inspect / replay`。完整契约见
[CLI surface design](./docs/superpowers/specs/2026-05-24-cli-surface-design.md)，
`.milkie/agents.json` manifest 约定见
[agent registration design](./docs/superpowers/specs/2026-05-24-agent-registration-design.md)。

---

## 核心概念：Agent = FSM

每个 Agent 由一组 **状态（state）** 描述。只有两种状态类型：

| 类型 | 行为 |
|------|------|
| `llm` | 循环调用 LLM。当工具触发 FSM 事件，或 LLM 输出纯文本（`DONE`）时退出。 |
| `action` | 执行确定性逻辑（如启动子 Agent），不调用 LLM。 |

状态通过 `on` 映射声明转移规则：

```typescript
fsm: {
  states: [
    {
      name: 'classify',
      type: 'llm',
      tools: ['classify_intent'],    // 工具触发 INTENT_ORDER 或 ESCALATE
      on: {
        INTENT_ORDER: 'collect_slots',
        ESCALATE:     'escalated',
      },
    },
    {
      name: 'collect_slots',
      type: 'llm',
      tools: ['collect_slot'],       // 槽位齐全后触发 SLOTS_COMPLETE
      on: { SLOTS_COMPLETE: 'confirm' },
    },
    {
      name:     'escalated',
      type:     'llm',
      terminal: true,                // 终止状态，无出口，输出最终消息
    },
  ],
}
```

工具通过 handler 中的 `ctx.emit()` 触发 FSM 转移：

```typescript
handler: async (input, ctx) => {
  const { intent } = input as { intent: string }
  ctx.emit(intent === 'order' ? 'INTENT_ORDER' : 'ESCALATE')
  return { intent }
}
```

多 Agent 编排、中断恢复、完整 API 参考，请查阅[使用指南](./docs/zh/guide.md)。

---

## 架构

milkie 由三个 peer 子系统构成：

- **Agent Runtime** — 执行引擎，将 LLM 驱动的自主性放入 FSM 结构中
- **Agent Trace** — 把每个 Agent run 保存为一等对象；通过 event log 的确定性投影提供 inspection / replay / fork / diff / lineage
- **Evolution** — 确定性实验子系统，用于迭代 Agent 配置

完整 target 架构、cross-cutting invariants、以及当前实现 vs. target 的
[Implementation Status](./ARCHITECTURE.md#implementation-status) 对账
都在 [ARCHITECTURE.md](./ARCHITECTURE.md) 中。

用户场景以 stories 形式跟踪，见 [docs/stories/](./docs/stories/)，
约定见 [README](./docs/stories/README.md)，索引见
[INDEX](./docs/stories/INDEX.md)。

---

## Examples

可跑的 demo 跟对应 story 配对放在 [examples/](./examples/) 下。每个 example
都有一份 SDK 脚本和一份等价 CLI 调用，跑在固化 fixture 上，**无需 API key**。

- [`s-005-replay`](./examples/s-005-replay/) — 确定性 replay（Phase 3）：
  用 in-process stub gateway 录一个 run，然后 replay 两遍（一次 SDK、
  一次 CLI），输出完全一致，**零真实 LLM 调用**。

---

## License

MIT
