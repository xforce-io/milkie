---
title: Phase 3 — Content-addressed cache + structural replay
date: 2026-05-24
status: approved
phase: 3
subsystems:
  - agent-trace
  - agent-runtime
unblocks:
  - s-005 (partial — byte-identical still needs Phase 4)
  - s-006 (cache foundation; fork primitive itself in Phase 5)
---

# Phase 3 — Content-addressed cache + structural replay

## 1. 目标与边界

Phase 3 将 Agent Trace 从 Phase 2 的 I/O event log 提升为 **Run 的事件流**。在这个模型里：

- **Run 是一等公民**：`runId` 是一个完整执行实例的聚合边界；一个 run 由 lifecycle events、LLM/tool I/O events 以及后续 phase 的 spawn/fork/diff 事件共同描述。
- **Event 是事实源**：event log 是 run 的唯一事实来源；cache、replay、fork、diff、lineage 都是基于 events 的 projection，不拥有独立事实。
- **Cache 与 replay 是 projection**：Phase 3 不引入新的 truth store，而是在读取某个 run 的 events 后构建临时 projection，用它驱动 structural replay。

在这个 Run/Event 模型之上，Phase 3 叠加两项能力：

1. **Content-addressed cache** — 对每个 LLM 请求与 tool 调用计算稳定 hash；hash → response 的映射作为 cache 索引。
2. **Structural replay** — `Milkie.replay(runId)` 重跑已记录的 run，所有 LLM/tool I/O 走 cache，不产生新的真调用。结果与原 run **状态等价**（status / lastTextOutput / 关键 domain 事件序列一致），但 **timestamps 与 UUIDs 不要求一致**（byte-identical 留到 Phase 4 的 non-determinism log）。

**显式不在范围内**（留后续 Phase）：

- ❌ 持久化的独立 cache store（cache 是 event log 的派生视图，重启即重建）
- ❌ Non-determinism log（clock/UUID/random 录制与回放）
- ❌ 含 spawn 的父 run 的 replay（其 event stream 含 `agent.spawn.started` → fail-fast）；纯叶 child run 不含 spawn 事件，允许 replay
- ❌ Fork primitive（在某事件分叉创建新 run）
- ❌ Structural diff（事件序列结构化对比）
- ❌ Lenient replay mode（cache miss 时 fallback 到真 LLM）

## 2. 设计决策（已 sign-off）

| 维度 | 选定方案 | 主要理由 |
|---|---|---|
| Scope | Cache + structural replay；不含 fork | 范围最小、风险最低、为 fork 铺路 |
| Cache key | 全量 `ModelRequest` 字段（model + messages + tools + system + temperature + 其他） + canonical JSON + SHA-256 | 严格保真；replay 不会被字段差异静默污染 |
| Tool replay | 同样 content-addressed：`hash(toolName + input)` | 与 LLM 同机制；副作用安全；fork 友好 |
| Divergence | Strict fail-fast：抛 `ReplayDivergenceError` | 早发现 bug；以后可在之上叠加 lenient mode |
| Run 边界 | `agent.run.started` / `agent.run.completed` 定义 run 的生命周期边界 | replay、fork、diff 都以 run 为聚合根，而不是以临时 runtime object 为入口 |
| Cache 存储 | `CacheIndex` 是 event log 的 projection（无独立 store） | 单一事实源；不引入新接口；fork 时按前缀 events 重建语义对称 |
| Lifecycle 事件 | 新增 `agent.run.started` / `agent.run.completed`，由 `RecordingIOPort` emit | run 自描述；replay 自包含（不依赖外部传 config）；AgentRuntime 不感知 EventStore |
| Spawn 事件 | 新增 `agent.spawn.started` / `agent.spawn.completed`；通过 `IIOPort.recordSpawn` 钩子由 RecordingIOPort emit | replay 才能识别父 run 含 spawn 并 fail-fast；IOPort 仍是 AgentRuntime 唯一 I/O 边界 |
| configSnapshot | 含完整 `availableTools[].schema + parallelSafe`（不只 name） | LLM request 含 tool schemas；schema 任一字段变 → 全量 hash miss；schema 必须随 run 自描述 |
| ReplayToolRegistry | 从 snapshot 还原 tool 集合，handler 全部 noop-throw | replay 不依赖宿主注册等价 tool；execute 永远不被触发（IOPort 在前拦截），noop 是兜底 |
| Tool error payload | structured `{ message, retryable?, code?, name? }` 替代 raw string | AgentRuntime 依赖 `err.retryable` 决定 retry；replay 必须保留这些字段才能等价重放 retry 行为 |
| Sub-agent | Phase 3 不支持 replay 含 spawn 的父 run，遇到 spawn 时 throw；叶 child run 可 replay | 减少 Phase 3 复杂度；嵌套 replay 与 fork 一起 Phase 5 做 |

## 3. 架构与数据流

```
[Run event stream: source of truth]          [Replay projections]
agent.run.started ──────┐
llm.requested (h1) ─────┤  Load Run by runId
llm.responded (h1, R1) ─┤  ──────────────────────────►  build projections
tool.requested (h2) ────┤                                   RunSnapshot
tool.responded (h2, O2)─┤                                   CacheIndex
… more I/O …            │                                       │
agent.run.completed ────┘                                       ▼
                                                  new AgentRuntime({
                                                    config: from RunSnapshot,
                                                    goal:   from RunSnapshot,
                                                    ioPort: new ReplayingIOPort(cacheIndex),
                                                    stateStore: ephemeral,
                                                    recorder: noop,
                                                  }).run()
                                                                │
                                              每次 invokeLLM/invokeTool：
                                                hash(call) → cacheIndex
                                                  hit  → return cached output
                                                  miss → throw ReplayDivergenceError
                                                                │
                                                                ▼
                                                  return AgentResult
                                                  （结构与原 run 等价）
```

## 4. 新增组件

### `src/trace/hash.ts`

```typescript
export function hashModelRequest(req: ModelRequest): string
export function hashToolCall(toolName: string, input: unknown): string
```

- canonical JSON：keys 排序后再 stringify（自实现 ~30 行，避免外部依赖；或引入 `json-stable-stringify`，由实现 plan 决定）
- SHA-256，返回 hex string（Node 内置 `crypto`）
- 全量字段：不忽略任何 ModelRequest 字段；对 undefined / null 区分处理（canonical 规则要明确）

### `src/trace/CacheIndex.ts`

```typescript
export class CacheIndex {
  static fromEvents(events: Event[]): CacheIndex
  /** Consume next queued response for this hash; throws if queue empty. */
  consumeLLM(hash: string): ModelResponse
  /** Consume next queued output (or rethrows reconstructed error). */
  consumeTool(hash: string): unknown
  /** Remaining (unconsumed) counts — should be zero after a successful replay. */
  remaining(): { llm: number, tool: number }
  allHashes(): { llm: string[], tool: string[] }
}
```

- `CacheIndex` 是纯 projection：只从 run events 构建，不写入、不持久化、不成为事实源
- 构建时按事件出现顺序入 FIFO 队列：`Map<hash, ModelResponse[]>` 与 `Map<hash, ToolOutcome[]>`
- **设计意图**：retry 循环或 FSM 多次循环可能让同 hash 重复出现且响应不同；FIFO 队列保留这个序列，replay 时按出现顺序 consume
- Tool error 重建为 Error 对象（含 retryable / code / name，见下文 §Tool error payload）
- Replay 结束时若 `remaining() !== 0` → `ReplayDivergenceError`（结构上不应剩响应）

### Run projection

Phase 3 不要求新增可持久化的 `Run` 表或独立 store，但实现上应把 `readByRunId(runId)` 读出的 events 先视为一个 run aggregate，再从中派生 replay 所需 projection：

- `RunSnapshot`：从 `agent.run.started` / `agent.run.completed` 提取 agentId、configSnapshot、goal、contextId、parentId、terminal status 等 lifecycle 信息
- `CacheIndex`：从 LLM/tool requested/responded 事件提取 replay I/O 队列
- 后续 Phase 的 fork/diff/lineage 也应沿用 `Run events -> Projection` 的模型，而不是绕过 event log 读取 trajectory 或 state store 作为事实源

### `src/trace/ReplayingIOPort.ts`

```typescript
export class ReplayingIOPort implements IIOPort {
  constructor(
    private readonly cache: CacheIndex,
    private readonly inner: IIOPort,  // 用于 now/uuid passthrough；不调 LLM/tool
  )
  // invokeLLM/invokeTool: 查 cache，miss 抛 ReplayDivergenceError；命中不调 inner
  // now/uuid: passthrough 到 inner
}
```

### `src/trace/ReplayDivergenceError.ts`

```typescript
export class ReplayDivergenceError extends Error {
  constructor(
    public readonly kind: 'llm' | 'tool',
    public readonly actualHash: string,
    public readonly summary: string,        // human-readable
    public readonly availableHashes: string[],  // 前 5 个
  )
}
```

### Event kinds 新增（`src/trace/types.ts`）

```typescript
| 'agent.run.started'      // payload: { agentId, configSnapshot, goal, contextId, parentId? }
| 'agent.run.completed'    // payload: { status: 'completed' | 'failed' | 'interrupted', lastTextOutput?, error? }
| 'agent.spawn.started'    // payload: { childRunId, childAgentId, taskId, parentRunId }
| 'agent.spawn.completed'  // payload: { childRunId, status }
```

`spawn.*` 事件由 IOPort 通过新增的 `recordSpawn` 钩子由 `RecordingIOPort` emit（见 §5）。Phase 3 不实现 sub-agent replay，但需要这两类事件出现在 event log 里，replay 才能 fail-fast 触发"unsupported"错误。

**configSnapshot 必须含 tool 完整 schema**。AgentRuntime 发 LLM 请求前会 `registry.toSchemas(tools)` 把 schema 嵌进 `ModelRequest`；schema 任何字段变了 → 全量 hash 变 → cache miss。因此 snapshot 形如：

```typescript
configSnapshot: {
  fsm:        { states, transitions, initial },
  prompts:    { ... },
  modelParams:{ ... },
  availableTools: Array<{
    name:         string,
    schema:       ToolSchema,   // 完整 schema（input/description）
    parallelSafe: boolean,       // 影响 parallel vs serial 分流
    // implementation 不进 snapshot
  }>,
}
```

Replay 时：构造 `ReplayToolRegistry`（见下）——从 snapshot 还原每个 tool，handler 是 noop（throw "should not be called during replay"）。这样 `registry.toSchemas()` 输出与原 run 字节一致 → LLM request hash 一致；noop handler 永远不被触发（`ReplayingIOPort.invokeTool` 在 AgentRuntime 调 execute 之前就拦截了），noop 是兜底防御。

### `src/trace/ReplayToolRegistry.ts`

```typescript
export class ReplayToolRegistry extends ToolRegistry {
  static fromSnapshot(availableTools: Array<{ name, schema, parallelSafe }>): ReplayToolRegistry
  // 复制 ToolRegistry 的 getForState / toSchemas 行为，handler 全部 noop-throw
}
```

存在的意义：让 replay 不依赖宿主 Milkie 注册了完全等价的 tool 列表，schema 完全从 snapshot 还原。

### Tool error payload 完整化

`ToolRespondedPayload.error` 字段从 `string` 升级为 structured：

```typescript
export interface ToolRespondedPayload {
  toolName: string
  output?:  unknown
  error?: {
    message:   string
    retryable?: boolean
    code?:      string
    name?:      string
  }
}
```

`RecordingIOPort` 在 catch 路径记录 `(err as { retryable?, code?, name? })` 这些可序列化字段。`CacheIndex.consumeTool` 命中 error 时重建 `Error(message)`，并把 retryable/code/name 直接挂在 Error 对象上（与 AgentRuntime.executeSingleTool 当前读法一致）。这样原 run 的 retry 行为在 replay 中等价复现。

### `Milkie.replay(runId, opts?)`

```typescript
async replay(runId: string): Promise<AgentResult>
// Phase 3: opts 留作扩展位但不接受任何选项
// 前置：Milkie 必须配置了 eventStore，并能找到该 runId
// 行为：构造 ReplayingIOPort + 一次性 AgentRuntime，跑完返回 result
// 限制：若 events 含 agent.spawn.started → throw ReplayError("sub-agent replay unsupported in Phase 3")
```

## 5. 对现有组件的改动

### `IIOPort`（src/runtime/IOPort.ts）

- 新增 `recordSpawn(event: 'started' | 'completed', payload): void`（or 拆成 `recordSpawnStarted` / `recordSpawnCompleted`，实现 plan 决定）
- `DefaultIOPort.recordSpawn`：no-op
- 让 IOPort 仍是 Agent Runtime 唯一的可观测边界——AgentRuntime 不直接接触 EventStore

### `RecordingIOPort`（src/trace/RecordingIOPort.ts）

- LLM/tool 事件 payload 增加 `requestHash: string` 字段（事件写入前用 `hashModelRequest` 算）
- 增加 `attach({ agentConfig, goal, contextId, parentId? })`：首次调用 invokeLLM/invokeTool 之前由 Milkie 调用，emit `agent.run.started`
- 增加 `detach({ status, lastTextOutput?, error? })`：run 结束时由 Milkie 调用，emit `agent.run.completed`
- 实现 `recordSpawn`：emit `agent.spawn.started` / `agent.spawn.completed` 事件

### `AgentRuntime`（src/runtime/AgentRuntime.ts）

- 在 `makeSubAgentTool` spawn 路径前后调用 `this.ioPort.recordSpawn('started', { childRunId, ... })` 与 `recordSpawn('completed', ...)`
- 现有 trajectory span（`agent.spawn`）保留，与 trace event 并存（trajectory 是 spans 的事实源、trace 是 run events 的事实源；§I8）

理由：lifecycle / spawn 事件由 IOPort 装饰层 emit，AgentRuntime 不感知 EventStore，IOPort 仍是它唯一的 I/O 边界。

### `Milkie`（src/runtime/Milkie.ts）

- `run()` / `runAgent()` 在 wrap RecordingIOPort 后立即调 `attach(...)`；结束后调 `detach(...)`
- 新增 `replay(runId)` 方法
- `replay` 内部构造的 IOPort 是裸的 ReplayingIOPort，**不经过 wrapIOPort 装饰链**（replay 不写新事件）

## 6. 不变式（测试断言）

- **I1**：每个可 replay run 必须有且仅有一个 `agent.run.started`；`agent.run.completed` 最多一个
- **I2**：`*.responded` 必须通过 `causedBy` 指向同一 run 内对应的 `*.requested`
- **I3**：`RecordingIOPort` 写入的 LLM 事件 payload 中 `requestHash === hashModelRequest(request)`
- **I4**：`hashModelRequest` 对等价输入稳定（同字段不同顺序的 object hash 相同）
- **I5**：`ReplayingIOPort.invokeLLM` 对 cache 命中的请求 **不调用 inner.invokeLLM**（mock 验证 call count = 0）
- **I6**：Strict mode 下 cache miss = throw（不允许静默 fallback）
- **I7**：Replay 期间不录新事件（EventStore append 不被调用）
- **I8**：CacheIndex / RunSnapshot 等 projection 不能反向修改 event log，event log 始终是唯一事实源
- **I9**：`agent.spawn.*` 事件的 `runId === parentRunId`（spawn 事件属父 run stream，不属 child run stream）；时间顺序上 spawn.started 必须先于对应 childRunId 的 `agent.run.started` emit
- **I10**：Tool error 在 replay 中 rethrow 的 Error 上 `retryable`/`code`/`name` 与原 run 一致（AgentRuntime retry 行为等价复现）

## 7. 边界情形

| 情形 | 处理 |
|---|---|
| 空 messages 的 LLM 请求 | hash 仍有效（[] 是合法输入），无特殊处理 |
| Tool 调用原 run 抛错 | tool.responded 携带 `error` → replay 时 `invokeTool` rethrow 等价 Error |
| Sub-agent spawn | replay 时检测到 `agent.spawn.started` 事件 → throw `ReplayError("sub-agent replay unsupported in Phase 3")` |
| Phase 2 录的 run（无 lifecycle 事件） | replay 检测到缺失 `agent.run.started` → throw `ReplayError("no lifecycle start event; run was recorded before Phase 3")` |
| 同一 hash 多次出现在 event log | CacheIndex 按事件顺序入 FIFO 队列；replay 时按调用顺序 consume，避免覆盖早期响应 |
| `replay(runId)` 找不到任何事件 | throw `ReplayError("no events for runId")` |
| Replay 期间 cache hit 但 cached response 缺字段 | throw `ReplayError("cached response malformed for hash X")` |

## 8. 测试策略

### Unit

| 文件 | 范围 |
|---|---|
| `src/__tests__/Hash.test.ts` | hash 稳定性 / canonical 排序 / 字段差异敏感性 |
| `src/__tests__/CacheIndex.test.ts` | FIFO 队列构建、同 hash 多响应按序 consume、empty events、tool error 重建（保留 retryable/code/name）、`remaining()` 报告未消费数 |
| `src/__tests__/ReplayingIOPort.test.ts` | hit/miss 行为、inner.invokeLLM 调用次数 = 0、divergence error 信息完整（kind / actualHash / availableHashes） |
| `src/__tests__/ReplayToolRegistry.test.ts` | 从 snapshot 还原后 `toSchemas()` 字节等价于原 registry；handler 调用必 throw "should not be called during replay" |
| `src/__tests__/Trace.test.ts`（扩展） | RecordingIOPort emit `agent.run.started/completed`、`agent.spawn.started/completed`；LLM/tool 事件 payload 含 `requestHash`；ToolResponded.error 是 structured |

### Integration

| 文件 | 范围 |
|---|---|
| `src/__tests__/Replay.test.ts` | 录 → replay → result 一致 + inner gateway 调用次数 = 0 |
| `src/__tests__/Replay.test.ts` | divergence: 改 prompt → replay → throws + 错误信息含 expected/actual hash |
| `src/__tests__/Replay.test.ts` | sub-agent: 父 run 含 spawn → replay → throws "unsupported in Phase 3" |
| `src/__tests__/Replay.test.ts` | 叶 child run（无 spawn 事件）→ replay 成功 |
| `src/__tests__/Replay.test.ts` | Tool retry 等价：原 run tool 抛 `{retryable: true}` 然后 retry 成功 → replay 行为字节等价（验证 I10）|
| `src/__tests__/Replay.test.ts` | 同 hash 多响应：FSM 多次循环触发相同 hash 不同响应 → replay 按序消费 |
| `src/__tests__/Replay.test.ts` | Phase 2 老 run: 录但无 lifecycle 事件 → replay → throws "no lifecycle start" |

### E2E

| 文件 | 范围 |
|---|---|
| `tests/e2e/s-005-deterministic-replay.e2e.test.ts`（新增） | 一个完整 case5/case6 风格的 run，record → replay → structural-equivalent；同步把 s-005 story status 从 draft 转 active |

### 关键断言模式

- ✅ 断言 `result.status === original.status && result.lastTextOutput === original.lastTextOutput`
- ✅ 断言 `inner.invokeLLM.mock.calls.length === 0`
- ✅ 断言事件序列（过滤 timestamp/uuid 后）等价
- ❌ 不断言 timestamps 一致
- ❌ 不断言 trajectory spans 一致

## 9. 落地次序与提交粒度

| # | Commit | 范围 |
|---|---|---|
| 1 | `feat(trace): add canonical hashing for LLM/tool requests` | `src/trace/hash.ts` + Hash.test.ts |
| 2 | `feat(trace): structured tool error payload + lifecycle/spawn event kinds` | `types.ts`（升级 ToolRespondedPayload.error；新增 agent.run.* 与 agent.spawn.* event kinds）+ Trace.test.ts 扩展 |
| 3 | `feat(runtime): IOPort.recordSpawn hook` | `IOPort.ts` 接口扩展 + DefaultIOPort no-op + AgentRuntime spawn 路径调用；trajectory span 保留并存 |
| 4 | `feat(trace): emit requestHash + lifecycle + spawn events in RecordingIOPort` | RecordingIOPort 实现 attach/detach/recordSpawn；Milkie.run/runAgent 调 attach/detach；事件 payload 含 requestHash；Trace.test.ts 扩展 |
| 5 | `feat(trace): add CacheIndex + ReplayingIOPort + ReplayToolRegistry` | CacheIndex（FIFO 队列）+ ReplayingIOPort + ReplayDivergenceError + ReplayToolRegistry + 各自 unit test |
| 6 | `feat(runtime): add Milkie.replay(runId) API` | Milkie.replay（含 RunSnapshot projection、sub-agent fail-fast 限制）+ Replay.test.ts integration |
| 7 | `test(e2e): add s-005 deterministic replay e2e test` | e2e + s-005 status: draft → active + INDEX.md |
| 8 | `docs: mark Phase 3 implemented in ARCHITECTURE.md` | ARCHITECTURE.md cache + replay 从 Target 移到 Implemented；更新 cross-cutting invariants（spawn events、structured error、Run/projection 模型） |

预期 ~750 行新代码 + ~480 行测试（含 ReplayToolRegistry、spawn events、structured error 多出的部分）。

## 10. Story readiness 变化

完成后：

- **s-005 (Deterministic replay)**：blocked → **partial**（cache ✓、replay engine ✓；仍需 non-determinism log 实现 byte-identical）
- **s-006 (Fork at event)**：blocked → blocked，但 cache 基础就绪；fork primitive 留 Phase 5
- INDEX.md 更新对应注释

## 11. 已知未决

设计层面已无未决，所有 sign-off 完成。以下细节留给 writing-plans 阶段决定，不会影响架构：

- canonical JSON 实现：自写 ~30 行 vs 引入 `json-stable-stringify`
- `recordSpawn` 签名：单一方法 + 'started'/'completed' 联合 vs 拆成两个方法
- `availableTools` 的 `schema` 字段类型：直接复用 `ToolSchema` 类型 vs 定义独立的 `SerializableToolSchema`

Phase 4+ 的接口形态留各自 phase 决定。
