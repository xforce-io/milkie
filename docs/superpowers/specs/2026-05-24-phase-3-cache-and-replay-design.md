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

Phase 3 在 Phase 2 的 event-sourced Agent Trace 之上叠加两项能力：

1. **Content-addressed cache** — 对每个 LLM 请求与 tool 调用计算稳定 hash；hash → response 的映射作为 cache 索引。
2. **Structural replay** — `Milkie.replay(runId)` 重跑已记录的 run，所有 LLM/tool I/O 走 cache，不产生新的真调用。结果与原 run **状态等价**（status / lastTextOutput / 关键 domain 事件序列一致），但 **timestamps 与 UUIDs 不要求一致**（byte-identical 留到 Phase 4 的 non-determinism log）。

**显式不在范围内**（留后续 Phase）：

- ❌ 持久化的独立 cache store（cache 是 event log 的派生视图，重启即重建）
- ❌ Non-determinism log（clock/UUID/random 录制与回放）
- ❌ Sub-agent / spawn 嵌套的 replay
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
| Cache 存储 | event log 的派生视图（无独立 store） | 单一事实源；不引入新接口；fork 时按前缀 events 重建语义对称 |
| Lifecycle 事件 | 新增 `agent.run.started` / `agent.run.completed`，由 `RecordingIOPort` emit | replay 自包含（不依赖外部传 config）；AgentRuntime 不感知 EventStore |
| Sub-agent | Phase 3 不支持 replay，遇到 spawn 时 throw | 减少 Phase 3 复杂度；Phase 5 与 fork 一起做 |

## 3. 架构与数据流

```
[已完成 run]                                  [replay]
agent.run.started ──────┐
llm.requested (h1) ─────┤  Read events for runId
llm.responded (h1, R1) ─┤  ──────────────────────────►  build CacheIndex
tool.requested (h2) ────┤                                   { h1 → R1,
tool.responded (h2, O2)─┤                                     h2 → O2, … }
… more I/O …            │                                       │
agent.run.completed ────┘                                       ▼
                                                  new AgentRuntime({
                                                    config: from started event,
                                                    goal:   from started event,
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
  getLLM(hash: string): ModelResponse | undefined
  getTool(hash: string): unknown | undefined   // returns output; throws for error-responded tools
  size(): { llm: number, tool: number }
  allHashes(): { llm: string[], tool: string[] }
}
```

- 纯函数构建；events 顺序无关（同 hash 的多次记录后值覆盖前值，记录 warning）
- Tool 的 error response 也存进 index，replay 时 rethrow（携带原 error message）

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
```

`configSnapshot` 是 `AgentConfig` 的 JSON-serializable 投影：FSM 定义（states / transitions）、tool 名引用（不含 implementation）、prompts、模型参数等。Tool implementations 在 replay 时不会被调用（走 cache），因此只需名引用；这意味着 replay 仍依赖 Milkie 注册了等价名字的 tool（或 ReplayingIOPort 不查 registry）。**决策**：ReplayingIOPort.invokeTool 不查 registry，只查 cache —— tool implementations 完全不参与 replay。这让 replay 与生产环境的 tool 实现解耦。

### `Milkie.replay(runId, opts?)`

```typescript
async replay(runId: string): Promise<AgentResult>
// Phase 3: opts 留作扩展位但不接受任何选项
// 前置：Milkie 必须配置了 eventStore，并能找到该 runId
// 行为：构造 ReplayingIOPort + 一次性 AgentRuntime，跑完返回 result
// 限制：若 events 含 agent.spawn.started → throw ReplayError("sub-agent replay unsupported in Phase 3")
```

## 5. 对现有组件的改动

### `RecordingIOPort`（src/trace/RecordingIOPort.ts）

- LLM/tool 事件 payload 增加 `requestHash: string` 字段（事件写入前用 `hashModelRequest` 算）
- 增加 `attach({ agentConfig, goal, contextId, parentId? })` 方法：在首次调用 invokeLLM/invokeTool 之前，由 Milkie 调用，emit `agent.run.started`
- 增加 `detach({ status, lastTextOutput?, error? })` 方法：run 结束时由 Milkie 调用，emit `agent.run.completed`

理由：lifecycle 事件由 IOPort 装饰层 emit，AgentRuntime 不感知 EventStore，IOPort 仍是它唯一的 I/O 边界。

### `Milkie`（src/runtime/Milkie.ts）

- `run()` / `runAgent()` 在 wrap RecordingIOPort 后立即调 `attach(...)`；结束后调 `detach(...)`
- 新增 `replay(runId)` 方法
- `replay` 内部构造的 IOPort 是裸的 ReplayingIOPort，**不经过 wrapIOPort 装饰链**（replay 不写新事件）

## 6. 不变式（测试断言）

- **I1**：`RecordingIOPort` 写入的 LLM 事件 payload 中 `requestHash === hashModelRequest(request)`
- **I2**：`hashModelRequest` 对等价输入稳定（同字段不同顺序的 object hash 相同）
- **I3**：`ReplayingIOPort.invokeLLM` 对 cache 命中的请求 **不调用 inner.invokeLLM**（mock 验证 call count = 0）
- **I4**：Strict mode 下 cache miss = throw（不允许静默 fallback）
- **I5**：Replay 期间不录新事件（EventStore append 不被调用）

## 7. 边界情形

| 情形 | 处理 |
|---|---|
| 空 messages 的 LLM 请求 | hash 仍有效（[] 是合法输入），无特殊处理 |
| Tool 调用原 run 抛错 | tool.responded 携带 `error` → replay 时 `invokeTool` rethrow 等价 Error |
| Sub-agent spawn | replay 时检测到 `agent.spawn.started` 事件 → throw `ReplayError("sub-agent replay unsupported in Phase 3")` |
| Phase 2 录的 run（无 lifecycle 事件） | replay 检测到缺失 `agent.run.started` → throw `ReplayError("no lifecycle start event; run was recorded before Phase 3")` |
| 同一 hash 多次出现在 event log | CacheIndex 取最后一次（覆盖语义）；不打日志（避免 stderr 噪声），通过 `CacheIndex.size()` 与 events 数量对照即可发现 |
| `replay(runId)` 找不到任何事件 | throw `ReplayError("no events for runId")` |
| Replay 期间 cache hit 但 cached response 缺字段 | throw `ReplayError("cached response malformed for hash X")` |

## 8. 测试策略

### Unit

| 文件 | 范围 |
|---|---|
| `src/__tests__/Hash.test.ts` | hash 稳定性 / canonical 排序 / 字段差异敏感性 |
| `src/__tests__/CacheIndex.test.ts` | 从 events 构建、duplicate hash、empty events、tool error 重建 |
| `src/__tests__/ReplayingIOPort.test.ts` | hit/miss 行为、inner 调用 0 次、divergence error 信息完整 |
| `src/__tests__/Trace.test.ts`（扩展） | RecordingIOPort emit lifecycle 事件、事件含 requestHash |

### Integration

| 文件 | 范围 |
|---|---|
| `src/__tests__/Replay.test.ts` | 录 → replay → result 一致 + inner gateway 调用次数 = 0 |
| `src/__tests__/Replay.test.ts` | divergence: 改 prompt → replay → throws + 错误信息含 expected/actual hash |
| `src/__tests__/Replay.test.ts` | sub-agent: 录含 spawn → replay → throws "unsupported in Phase 3" |
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
| 2 | `feat(trace): emit requestHash + lifecycle events in RecordingIOPort` | types.ts + RecordingIOPort + Milkie attach/detach + Trace.test.ts 扩展 |
| 3 | `feat(trace): add CacheIndex + ReplayingIOPort` | CacheIndex + ReplayingIOPort + ReplayDivergenceError + 各自 unit test |
| 4 | `feat(runtime): add Milkie.replay(runId) API` | Milkie.replay + Replay.test.ts integration |
| 5 | `test(e2e): add s-005 deterministic replay e2e test` | e2e + s-005 status: draft → active + INDEX.md |
| 6 | `docs: mark Phase 3 implemented in ARCHITECTURE.md` | ARCHITECTURE.md cache + replay 从 Target 移到 Implemented；更新 cross-cutting invariants |

预期 ~600 行新代码 + ~400 行测试。

## 10. Story readiness 变化

完成后：

- **s-005 (Deterministic replay)**：blocked → **partial**（cache ✓、replay engine ✓；仍需 non-determinism log 实现 byte-identical）
- **s-006 (Fork at event)**：blocked → blocked，但 cache 基础就绪；fork primitive 留 Phase 5
- INDEX.md 更新对应注释

## 11. 已知未决

无。所有 Phase 3 设计决策已 sign-off。Phase 4+ 的接口形态留各自 phase 决定。
