---
title: Phase 4 — 非确定性日志（byte-identical 重放）设计
status: approved
created: 2026-05-24
supersedes: -
related:
  - docs/superpowers/specs/2026-05-24-phase-3-cache-and-replay-design.md
  - docs/stories/s-005-deterministic-replay.md
  - roadmap.md#phase-4
---

# Phase 4 — 非确定性日志（byte-identical 重放）设计

## 1. 目标与边界

Phase 3 落地了**结构等价**重放——`status` / `lastTextOutput` 与原始 run 一致，但 agent 内部消费的 timestamp / UUID 在重放时被**重新采样**。一旦 agent 把这些值嵌入下游（比如把 `port.uuid()` 生成的 trace id 放进 LLM prompt 或 tool input），重新采样的值会改变下游请求的内容哈希 → cache miss → `ReplayDivergenceError`。

Phase 4 把重放升级为 **byte-identical**：

- 录制时，`RecordingIOPort` 把 agent 对 `port.now()` / `port.uuid()` 的每次调用写入事件日志
- 重放时，`ReplayingIOPort` 从事件日志按 FIFO 顺序取出并返回这些值
- 任何顺序/数量不匹配立即抛 `ReplayDivergenceError`

### 边界（明确不做的）

- **`random.consumed`**：`src/` 里目前零调用点。Y' 决策：等真有依赖再扩
- **Trajectory 层直接 `Date.now()` 调用**：在 `src/trajectory/*.ts` 里旁路了 IOPort。TrajectoryStore 在 roadmap 已标待退役，本次不掺
- **`SQLiteStore` TTL 用的 `Date.now()`**：运行时基础设施，不属于 agent trace 语义
- **Per-operator side-effect 重放策略**：roadmap 提的"哪些工具重放时打 live"是 Phase 5 fork 的关注点。Phase 4 默认 = 全部 from cache（与现有重放行为一致）
- **事件 schema 版本号**：跨阶段隐性约定，本次不引入
- **`trace inspect` 对 nondet 事件的默认过滤**：等真有人觉得吵再加，不预设需求

## 2. 设计决策（已 sign-off）

| 决策 | 选择 | 理由 |
|---|---|---|
| 记录颗粒度 | A：每次 `now()` / `uuid()` 调用一条独立事件 | 复用现有 append-only 事件流；trace 工具天然覆盖；事件数虽大但单条小、可流式 |
| 事件类型切分 | Y'：新增 `clock.read` + `uuid.generated` 两个 `EventKind`，**不**加 `random.consumed` | 与 `llm.requested` / `tool.requested` 的 "一件事一个 type" 风格一致；YAGNI |
| 重放严格度 | P-wide：过消费立即抛 `ReplayDivergenceError`；replay 尾部检查所有四个队列（clock / uuid / llm / tool）剩余非零时也抛 | fail-fast 是正解；未上线无需向后兼容 |
| 录制 sync/async 阻抗 | 保持 `IIOPort.now()` / `.uuid()` 同步签名 + RecordingIOPort 内部 pending buffer + 在每个 async 方法入口 flush | 避免改 IIOPort 接口引发全栈 async 染色 |
| s-005 fixture | 重新录制（删旧 + 跑 `record.ts`） | 简化 > 兼容；未上线 |
| s-002 fixture | 重新录制；HTML report 自然展示 nondet 事件 | report 的价值 = "看到 agent 真实做了什么"，nondet 也是真实 |
| Side-effect 策略 | 全部 from cache；per-operator hook 设计延期到 Phase 5 fork | Phase 4 范围克制；fork 落地时再统一设计 hook |

## 3. 架构与数据流

```
Record path
═══════════
agent code
    └→ port.now()    (sync)
       RecordingIOPort.now()
         val = inner.now()              // = Date.now()
         pendingNondet.push({clock, val})
         return val                     // ← agent 拿到值
    └→ port.invokeLLM(req)    (async)
       RecordingIOPort.invokeLLM()
         await flushPendingNondet()      // 把 pending 按入队顺序逐条 store.append
         hash = hashModelRequest(req)
         await store.append({llm.requested, ...})
         resp = await inner.invokeLLM(req)
         await store.append({llm.responded, ..., causedBy})
         return resp

Replay path
═══════════
agent code
    └→ port.now()    (sync)
       ReplayingIOPort.now()
         return cache.consumeClock()    // 队列空 → ReplayDivergenceError
    └→ port.invokeLLM(req)    (async)
       ReplayingIOPort.invokeLLM()
         hash = hashModelRequest(req)
         return cache.consumeLLM(hash)  // 未命中 → ReplayDivergenceError
Milkie.replay() 尾部
       cache.clockRemaining() / uuidRemaining() / llmRemaining() / toolRemaining()
       任一非零 → ReplayDivergenceError
```

**关键不变量**：
- `RecordingIOPort.now()` / `.uuid()` 返回值前，对应 nondet 事件已进入 pending buffer
- 下次该 IOPort 实例的任一 async 方法被 await 完成时，pending buffer 中的所有事件**已按入队顺序**写入 store
- `detach()` 是兜底 flush，保证 run 结束时 buffer 为空
- 重放期间 `ReplayingIOPort` 从不调用 `inner.now()` / `inner.uuid()`——任何对 inner 的调用都是 bug 信号

## 4. 新增 / 修改组件

### `src/trace/types.ts` — 事件类型新增

```typescript
export type EventKind =
  | 'llm.requested' | 'llm.responded'
  | 'tool.requested' | 'tool.responded'
  | 'agent.run.started' | 'agent.run.completed'
  | 'clock.read'         // ← 新增
  | 'uuid.generated'     // ← 新增

export interface ClockReadPayload    { value: number }
export interface UuidGeneratedPayload { value: string }

export type ClockReadEvent     = Event<ClockReadPayload>     & { type: 'clock.read' }
export type UuidGeneratedEvent = Event<UuidGeneratedPayload> & { type: 'uuid.generated' }
```

加进 `AnyEvent` union 末尾。

### `src/trace/RecordingIOPort.ts` — 真录制

新增字段：`private pendingNondet: Array<ClockReadPayload | UuidGeneratedPayload & { _kind: 'clock' | 'uuid' }> = []`（或更清晰用 sum type）。

`now()` 同步实现：
```typescript
now(): number {
  const val = this.inner.now()
  this.pendingNondet.push({ kind: 'clock', value: val })
  return val
}
```

`uuid()` 同结构。

新增私有方法 `flushPendingNondet(): Promise<void>`：按入队顺序逐条 `store.append({ id: inner.uuid(), runId, type: 'clock.read'|'uuid.generated', actor, timestamp: inner.now(), payload })` 然后清空 pending。

**关键**：写 nondet 事件自身的 `id` 和 `timestamp` 字段时**直接调 `inner.uuid()` / `inner.now()`**，**不**经过 pending buffer——否则递归。这是 "infrastructure 用 vs agent 用" 的具体落地。

每个 async 方法（`attach` / `detach` / `invokeLLM` / `invokeTool`）开头插入：
```typescript
await this.flushPendingNondet()
```

`detach()` 完成 buffer flush 后再 append `agent.run.completed`，保证 detach 后 buffer 严格为空。

### `src/trace/CacheIndex.ts` — 新增队列

新增字段：
```typescript
private clockQueue: number[] = []
private uuidQueue: string[] = []
private clockConsumed = 0        // 用于错误诊断
private uuidConsumed = 0
```

`fromEvents` 已有的循环里追加：
```typescript
if (event.type === 'clock.read')      this.clockQueue.push((event.payload as ClockReadPayload).value)
if (event.type === 'uuid.generated')  this.uuidQueue.push((event.payload as UuidGeneratedPayload).value)
```

新增公开方法：
```typescript
consumeClock(): number     // 队列空 → throw new CacheIndexEmptyError('clock')
consumeUuid(): string      // 队列空 → throw new CacheIndexEmptyError('uuid')
clockRemaining(): number   // 用于 replay 尾部欠消费检查
uuidRemaining(): number
llmRemaining(): number     // 已有结构补一下 accessor，P-wide 需要
toolRemaining(): number    // 同上
```

`CacheIndexEmptyError` 已存在，扩 kind 联合类型加 `'clock' | 'uuid'`。

### `src/trace/ReplayingIOPort.ts` — 真重放

`now()` / `uuid()` 改为：
```typescript
now(): number {
  try { return this.cache.consumeClock() }
  catch (err) {
    if (err instanceof CacheIndexEmptyError)
      throw new ReplayDivergenceError('clock', '', `clock.read queue exhausted after ${this.cache.clockConsumed} consumed`, [])
    throw err
  }
}
// uuid() 同结构
```

`inner` 字段保留（LLM/tool 路径仍可能需要）但 now/uuid 不再走 inner。

### `src/trace/ReplayDivergenceError.ts` — kind 扩展

```typescript
export type DivergenceKind = 'llm' | 'tool' | 'clock' | 'uuid'
```

对 `clock` / `uuid` kind，`hash` 字段传 `''`，`expectedHashes` 字段传 `[]`，诊断信息全部走 `message`。

### `src/runtime/Milkie.ts` — replay 尾部检查

`Milkie.replay()` 在 `runtime.run()` 返回之后、`return result` 之前：

```typescript
const r = result
if (divergenceError) throw divergenceError

// P-wide: 四个队列任一非空都算 divergence
const remaining = {
  clock: cache.clockRemaining(),
  uuid:  cache.uuidRemaining(),
  llm:   cache.llmRemaining(),
  tool:  cache.toolRemaining(),
}
for (const [kind, n] of Object.entries(remaining)) {
  if (n > 0) throw new ReplayDivergenceError(
    kind as DivergenceKind, '',
    `${n} ${kind} event(s) unconsumed after replay completed`, [])
}
return r
```

## 5. 不变式（测试用）

1. **录制完整性**：录制结束后，事件日志中 `clock.read` 事件数 = 该 run 中 `port.now()` 被调用次数；`uuid.generated` 同理
2. **录制顺序**：clock.read[i] 在事件日志中的位置 < 同 run 内任何在 port.now() 第 i 次返回之后写入的事件
3. **重放确定性**：同一份事件日志，N 次 replay 产生 N 个完全相同的 `AgentResult`
4. **byte-identical**：替换底层 `inner` 的 `now()` / `uuid()` 为 throw 函数，replay 仍然成功（证明 replay 不依赖 inner）
5. **过消费检测**：重放路径多调一次 `port.now()` → 立即抛 `ReplayDivergenceError` kind='clock'
6. **欠消费检测**：重放路径少调任一 IO → `Milkie.replay()` 返回前抛 `ReplayDivergenceError`
7. **infrastructure 不录**：RecordingIOPort 写自己事件的 `timestamp` / `id` 不会触发递归 nondet 事件（事件日志中没有"为录 clock.read 而生成的 clock.read"）

## 6. 边界情形

- **空 run**：agent 启动失败、attach 前就 detach。pending buffer 可能含 attach 前调用的 nondet（构造 AgentRuntime 时的 contextId/agentRunId）。detach 仍然 flush。
- **interrupt 路径**：agent 被中断，detach 还是会执行。pending buffer flush 正常。
- **sub-agent**：每个 sub-agent 有独立 runId，事件写到自己的 `<subAgentRunId>.jsonl`，CacheIndex 实例也是每 run 独立——nondet 队列天然隔离。
- **`RecordingIOPort` 实例并发**：当前架构假设单 agent 串行，pending buffer 无锁。如果未来支持单 IOPort 实例多 agent 并发，buffer 需要重新设计（不在 Phase 4 范围）。
- **`Date.now()` 返回非单调**（系统时钟回拨）：记什么就重放什么。byte-identical 不依赖单调性。

## 7. 测试策略

### Unit

- `RecordingIOPort.nondet.test.ts`：now/uuid 写 pending；flushPendingNondet 顺序 + 清空；async 方法入口自动 flush；detach 兜底 flush
- `ReplayingIOPort.nondet.test.ts`：now/uuid 从 cache 取；队列空抛 ReplayDivergenceError；inner.now/uuid 永不被调用
- `CacheIndex.nondet.test.ts`：fromEvents 入队；consumeClock/Uuid FIFO；clockRemaining/uuidRemaining/llmRemaining/toolRemaining 准确

### Integration

- `Replay.nondet.test.ts`（核心）：
  - **byte-identical 证明**：agent 把 `port.uuid()` 生成的 trace_id 嵌入 LLM 请求 messages。无 nondet 重放：cache miss → fail。有 nondet 重放：cache 命中 → success
  - **inner 隔离**：替换 `inner.now()` 为 `() => { throw new Error('never') }`、`inner.uuid()` 同。replay 仍成功
  - **过消费**：构造录制有 3 个 clock.read，重放路径调 4 次 → 第 4 次抛 ReplayDivergenceError kind='clock'
  - **欠消费**：构造录制有 3 个 uuid.generated，重放路径只调 2 次 → Milkie.replay() 尾部抛 ReplayDivergenceError kind='uuid'
  - **欠消费 P-wide 对 llm/tool**：录制 1 个 llm.requested 但重放 agent 跳过该 LLM 调用 → 尾部抛 ReplayDivergenceError kind='llm'

### E2E

- `tests/e2e/s-005-deterministic-replay.e2e.test.ts`：
  - 删除现有 fixture，重新跑 record.ts 生成新 fixture（带 nondet 事件）
  - 断言升级：原来只比 `status` / `output`，现在加 **inner 隔离断言**——重放时给 ReplayingIOPort 的 inner 装 throw 函数，replay 仍成功
  - 旧文件 `examples/s-005-replay/.milkie/last-run.txt` 和 `.milkie/runs/ef2a8183-….jsonl` 删除并替换

## 8. 落地次序

按 TDD 节奏，每步都是失败测试 → 实现 → 通过 → 提交：

1. **schema**：types.ts 加 `clock.read` / `uuid.generated` + payload + Event aliases
2. **CacheIndex 扩展**：clock/uuid 队列 + consume/Remaining accessor + llm/tool Remaining accessor（P-wide 用）
3. **CacheIndexEmptyError + ReplayDivergenceError**：kind 联合扩展
4. **RecordingIOPort 真录制**：pending buffer + flushPendingNondet + async 方法入口插入 flush + detach 兜底
5. **ReplayingIOPort 真重放**：now/uuid 走 cache，inner 不再被 now/uuid 触碰
6. **Milkie.replay() 尾部欠消费检查**：四队列检查
7. **Integration test `Replay.nondet.test.ts`**：覆盖第 7 节核心场景
8. **s-005 fixture 重录 + e2e 断言升级**
9. **s-002 fixture 重录**：跑 examples/s-002-inspect/record.ts，更新 `.milkie/runs/` + `last-run.txt`
10. **roadmap.md 更新**：Phase 4 标 completed；side-effect 策略条目改 "Phase 5 prerequisite"

## 9. roadmap 影响

完成后 `roadmap.md`：

- TL;DR "Next big rock: Phase 4" → "Completed: Phase 1–4"
- "Completed (Phase 1–4)" 新增 Phase 4 段
- "Up next" 删 Phase 4 段，Phase 5 提前
- "Open architectural questions" 里 "Replay side-effect policy" 改为 "Phase 5 prerequisite — Phase 4 declared all-from-cache; per-operator hook 与 fork 一起设计"
- s-005 INDEX 状态：active（已是 active，断言升级即可）
