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

Phase 3 在 Phase 2 的 I/O event log 之上建立 **Run/Event 模型**：

- **Run 是一等公民**：`runId` 是一个完整执行实例的聚合边界。
- **Event 是 run 的事实源**：event log 是该 run 的 **历史 I/O 与 lifecycle** 的唯一事实来源。
- **Cache 与 replay 是 projection**：从 events 派生的临时投影，不引入新的 truth store。

在这个模型上叠加两项能力：

1. **Content-addressed cache** — 对每个 LLM 请求与 tool 调用计算稳定 hash；hash → response 的映射作为 cache 索引。
2. **Structural replay** — `Milkie.replay(runId)` 重跑已记录的 run，所有 LLM/tool I/O 走 cache，不产生新的真调用。结果与原 run **状态等价**（status / lastTextOutput / 关键 domain 事件序列一致），**timestamps 与 UUIDs 不要求一致**（byte-identical 留到 Phase 4 的 non-determinism log）。

### Run/Event 事实源的范围（重要边界）

Phase 3 的 event log 携带的是 **lifecycle identity + I/O 历史**，**不是** portable execution spec：

- lifecycle event 只存 agent 身份（`agentId` + run 入口参数）
- replay 时由当前 Milkie 注册表提供 `AgentConfig` 与 tool 实现
- replay 成功要求宿主环境的 agent / tool 定义与原 run 结构等价

这是 same-host / same-code / same-registered-agent replay。Portable replay（含完整 config 序列化）留给后续 phase 在有具体场景时再加。

**显式不在范围内**：

- ❌ 持久化的独立 cache store（cache 是 event log 的派生视图，重启即重建）
- ❌ Non-determinism log（clock/UUID/random 录制与回放）
- ❌ Portable replay：跨进程导出 trace 后在别处 replay
- ❌ Long-term replay across config evolution：注册的 agent/tool 与原 run 已不等价时的 replay
- ❌ Sub-agent-aware replay：含 sub-agent 的 run 在 Phase 3 可能以 cache miss / divergence 失败、错误信息不保证精确（留 Phase 5）
- ❌ Fork primitive（在某事件分叉创建新 run）
- ❌ Structural diff（事件序列结构化对比）
- ❌ Lenient replay mode（cache miss 时 fallback 到真 LLM）

## 2. 设计决策（已 sign-off）

| 维度 | 选定方案 | 主要理由 |
|---|---|---|
| Scope | Cache + structural replay；不含 fork、不含 spawn 事件 | 范围最小、价值密度最高、为 fork 铺路 |
| Cache key | 全量 `ModelRequest` 字段 + canonical JSON + SHA-256 | 严格保真；replay 不会被字段差异静默污染 |
| Tool replay | 同样 content-addressed：`hash(toolName + input)` | 与 LLM 同机制；副作用安全；fork 友好 |
| Divergence | Strict fail-fast：抛 `ReplayDivergenceError` | 早发现 bug；以后可在之上叠加 lenient mode |
| Run 边界 | `agent.run.started` / `agent.run.completed` 定义 run lifecycle | replay、fork、diff 以 run 为聚合根 |
| Lifecycle payload | `agent.run.started` 只存 `{ agentId, goal, input, contextId, parentId? }`，**不含** `configSnapshot` | event log = lifecycle identity + I/O 历史，**不是** portable execution spec |
| Cache 存储 | `CacheIndex` 是 event log 的 projection（无独立 store、无独立类继承层） | 单一事实源；fork 时按前缀 events 重建语义对称 |
| Tool 在 replay 中 | 直接复用当前 Milkie 注册的 tool registry；`ReplayingIOPort.invokeTool` 在 AgentRuntime 调 execute 之前拦截 | 不需要 stub registry；execute 被调到 = ReplayingIOPort 漏挡 = bug |
| Tool error payload | structured `{ message, retryable?, code?, name? }` 替代 raw string | AgentRuntime 依赖 `err.retryable` 决定 retry；replay 必须保留这些字段才能等价重放 retry 行为 |
| Sub-agent | Phase 3 不主动检测、不友好报错；含 spawn 的 run 可能 cache miss 失败，错误不精确 | 避免为"友好错误"建一整套 spawn 事件机制；真正的 sub-agent replay 在 Phase 5 与 fork 一起做 |

## 3. 架构与数据流

```
[Run event stream: source of truth]          [Replay projections]
agent.run.started ──────┐
llm.requested (h1) ─────┤  Load Run by runId
llm.responded (h1, R1) ─┤  ──────────────────────────►  build projections
tool.requested (h2) ────┤                                   RunSnapshot { agentId, goal, input, ... }
tool.responded (h2, O2)─┤                                   CacheIndex { h1 → R1 队列, h2 → O2 队列 }
… more I/O …            │                                       │
agent.run.completed ────┘                                       ▼
                                              Milkie.replay(runId):
                                                lookup agents.get(agentId)  // 复用宿主注册的 AgentConfig
                                                new AgentRuntime({
                                                  config:     fromRegistry,
                                                  goal:       fromSnapshot,
                                                  input:      fromSnapshot,
                                                  ioPort:     new ReplayingIOPort(cacheIndex, inner),
                                                  stateStore: ephemeral,
                                                  recorder:   noop,
                                                }).run(input)
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

### `RunSnapshot`

从 events 派生的简单数据结构，**不需要独立类**——一个函数即可：

```typescript
export function extractRunSnapshot(events: Event[]): {
  agentId: string
  goal:    string
  input?:  unknown
  contextId: string
  parentId?: string
  terminalStatus?: 'completed' | 'error' | 'interrupted'
}
```

实现：从 events 找唯一的 `agent.run.started` 读 payload；从 `agent.run.completed`（若有）读 terminalStatus。

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
| 'agent.run.started'      // payload: { agentId, goal, input?, contextId, parentId? }
| 'agent.run.completed'    // payload: { status: 'completed' | 'error' | 'interrupted', lastTextOutput?, error? }
```

`agent.run.started.payload` 只携带 **lifecycle identity** 与 **run 入口参数**——不携带 `AgentConfig`、不携带 tool schemas、不携带 prompts。`Milkie.replay()` 用 `agentId` 从当前注册表查到 `AgentConfig`，复用当前 tool registry。

**约定**：replay 假定宿主 Milkie 已注册同名 agent，且注册的 AgentConfig 与 tool 集合与原 run 结构等价（否则 LLM request hash 会变 → divergence）。文档点出，不在代码里检测——检测复杂度不亚于序列化 config。

### Tool error payload 完整化

`ToolRespondedPayload.error` 字段从 `string` 升级为 structured：

```typescript
export interface ToolRespondedPayload {
  toolName: string
  output?:  unknown
  error?: {
    message:    string
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
// 前置：Milkie 必须配置了 eventStore；该 runId 必须存在
// 前置：Milkie 已注册同名 agentId（从 events 读出）
// 行为：
//   1. eventStore.readByRunId(runId) → events
//   2. extractRunSnapshot(events) → { agentId, goal, input, contextId, parentId? }
//   3. CacheIndex.fromEvents(events) → cacheIndex
//   4. 从 this.agents 拿到 AgentConfig
//   5. 构造一次性 AgentRuntime（ioPort = ReplayingIOPort，recorder = noop，stateStore = MemoryStore）
//   6. runtime.run(input) → AgentResult
// Phase 3: opts 留作扩展位但不接受任何选项
```

## 5. 对现有组件的改动

### `RecordingIOPort`（src/trace/RecordingIOPort.ts）

- LLM/tool 事件 payload 增加 `requestHash: string` 字段（事件写入前用 `hashModelRequest` / `hashToolCall` 算）
- 增加 `attach({ agentId, goal, input?, contextId, parentId? })`：首次调用 invokeLLM/invokeTool 之前由 Milkie 调用，emit `agent.run.started`
- 增加 `detach({ status, lastTextOutput?, error? })`：run 结束时由 Milkie 调用，emit `agent.run.completed`
- Tool error 写入 structured payload（见 §Tool error payload）

### `Milkie`（src/runtime/Milkie.ts）

- `invoke()` 在 wrap RecordingIOPort 后立即调 `attach(...)`；结束后（finally）调 `detach(...)`
- `resume()` 的 record/replay 语义不纳入 Phase 3；后续与 checkpoint / non-determinism log 一起设计
- 新增 `replay(runId)` 方法（见 §4）
- `replay` 内部构造的 IOPort 是裸的 ReplayingIOPort，**不经过 wrapIOPort 装饰链**（replay 不写新事件）

### `AgentRuntime`

无改动。AgentRuntime 不感知 EventStore，所有事件都由 RecordingIOPort 在 I/O 边界 emit；lifecycle 事件由 Milkie 触发 attach/detach。

## 6. 不变式（测试断言）

- **I1**：每个可 replay run 必须有且仅有一个 `agent.run.started`；`agent.run.completed` 最多一个
- **I2**：`*.responded` 必须通过 `causedBy` 指向同一 run 内对应的 `*.requested`
- **I3**：`RecordingIOPort` 写入的 LLM 事件 payload 中 `requestHash === hashModelRequest(request)`
- **I4**：`hashModelRequest` 对等价输入稳定（同字段不同顺序的 object hash 相同）
- **I5**：`ReplayingIOPort.invokeLLM` 对 cache 命中的请求 **不调用 inner.invokeLLM**（mock 验证 call count = 0）
- **I6**：Strict mode 下 cache miss = throw（不允许静默 fallback）
- **I7**：Replay 期间不录新事件（EventStore append 不被调用）
- **I8**：Tool error 在 replay 中 rethrow 的 Error 上 `retryable`/`code`/`name` 与原 run 一致（AgentRuntime retry 行为等价复现）

## 7. 边界情形

| 情形 | 处理 |
|---|---|
| 空 messages 的 LLM 请求 | hash 仍有效（[] 是合法输入），无特殊处理 |
| Tool 调用原 run 抛错 | tool.responded 携带 structured error → replay 时 `consumeTool` rethrow 等价 Error |
| Sub-agent 含 spawn 的 run replay | Phase 3 不主动检测；可能正常完成（叶 agent 也走 cache）也可能 cache miss / divergence；错误信息不保证精确——文档明示，留 Phase 5 |
| Phase 2 录的 run（无 lifecycle 事件） | replay 检测到缺失 `agent.run.started` → throw `ReplayError("no lifecycle start event; run was recorded before Phase 3")` |
| 同一 hash 多次出现在 event log | CacheIndex 按事件顺序入 FIFO 队列；replay 时按调用顺序 consume，避免覆盖早期响应 |
| `replay(runId)` 找不到任何事件 | throw `ReplayError("no events for runId")` |
| `replay(runId)` 找到 events 但 Milkie 未注册该 agentId | throw `ReplayError("agentId X not registered on this Milkie instance")` |
| Replay 期间 cache hit 但 cached response 缺字段 | throw `ReplayError("cached response malformed for hash X")` |

## 8. 测试策略

### Unit

| 文件 | 范围 |
|---|---|
| `src/__tests__/Hash.test.ts` | hash 稳定性 / canonical 排序 / 字段差异敏感性 |
| `src/__tests__/CacheIndex.test.ts` | FIFO 队列构建、同 hash 多响应按序 consume、empty events、tool error 重建（保留 retryable/code/name）、`remaining()` 报告未消费数 |
| `src/__tests__/ReplayingIOPort.test.ts` | hit/miss 行为、inner.invokeLLM 调用次数 = 0、divergence error 信息完整（kind / actualHash / availableHashes） |
| `src/__tests__/Trace.test.ts`（扩展） | RecordingIOPort emit `agent.run.started/completed` 含正确 payload；LLM/tool 事件 payload 含 `requestHash`；ToolResponded.error 是 structured |

### Integration

| 文件 | 范围 |
|---|---|
| `src/__tests__/Replay.test.ts` | 录 → replay → result 一致 + inner gateway 调用次数 = 0 |
| `src/__tests__/Replay.test.ts` | divergence: 改 prompt → replay → throws + 错误信息含 actualHash / availableHashes |
| `src/__tests__/Replay.test.ts` | Tool retry 等价：原 run tool 抛 `{retryable: true}` 然后 retry 成功 → replay 行为字节等价（验证 I8）|
| `src/__tests__/Replay.test.ts` | 同 hash 多响应：FSM 多次循环触发相同 hash 不同响应 → replay 按序消费 |
| `src/__tests__/Replay.test.ts` | Phase 2 老 run: 录但无 lifecycle 事件 → replay → throws "no lifecycle start" |
| `src/__tests__/Replay.test.ts` | agentId 未注册：replay → throws "agentId X not registered" |

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
| 2 | `feat(trace): structured tool error + lifecycle event kinds + requestHash on I/O events` | `types.ts`（升级 ToolRespondedPayload.error；新增 agent.run.started/completed event kinds；I/O event payload 加 requestHash）+ RecordingIOPort 实现 attach/detach + Milkie.run/runAgent 调 attach/detach + Trace.test.ts 扩展 |
| 3 | `feat(trace): add CacheIndex + ReplayingIOPort + ReplayDivergenceError` | CacheIndex（FIFO 队列）+ ReplayingIOPort + ReplayDivergenceError + extractRunSnapshot + 各自 unit test |
| 4 | `feat(runtime): add Milkie.replay(runId) API` | Milkie.replay + Replay.test.ts integration |
| 5 | `test(e2e): add s-005 deterministic replay e2e test + ARCHITECTURE.md update` | e2e + s-005 status: draft → active + INDEX.md 更新 + ARCHITECTURE.md 把 cache + replay 从 Target 移到 Implemented |

预期 ~450 行新代码 + ~350 行测试。

## 10. Story readiness 变化

完成后：

- **s-005 (Deterministic replay)**：blocked → **partial**（cache ✓、replay engine ✓；仍需 non-determinism log 实现 byte-identical）
- **s-006 (Fork at event)**：blocked → blocked，但 cache 基础就绪；fork primitive 留 Phase 5
- INDEX.md 更新对应注释

## 11. 已知未决

设计层面已无未决，所有 sign-off 完成。以下细节留给 writing-plans 阶段决定，不会影响架构：

- canonical JSON 实现：自写 ~30 行 vs 引入 `json-stable-stringify`
- `agent.run.started.payload.input` 字段类型：`unknown` 还是更严格

Phase 4+ 的接口形态留各自 phase 决定。
