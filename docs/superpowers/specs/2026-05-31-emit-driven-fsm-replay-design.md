# #60 设计:emit 驱动的 FSM 转移可 replay

状态:已认可,进入实现
关联:follow-up of #73(PR #74,`wm.mutated`);Phase 5(fork/diff/suite)前置

## 背景与根因

`replay()` 的本质:`ReplayingIOPort.invokeTool` 直接吐缓存输出、**不跑 handler**(忽略 `execute`
thunk)。于是 handler 的副作用全丢。

- **#73 已解的 WM 副作用**:`executeSingleTool` 每次工具调用录一条 `wm.mutated` 全量快照;replay 时
  `replayWmSnapshots.shift()` 把快照灌回 `this.memory`(positional FIFO 对齐)。
- **#60 剩的 emit 副作用**:handler 里 `ctx.emit` → `fsm.emitEvent` 设 `pendingEvent`。replay 不跑
  handler → `pendingEvent` 恒为 null → `runLLMState` 的 `processPendingEvent()` 返回 null →
  `continue` → 多发一次未录的 LLM 调用 → `ReplayDivergenceError(llm)`。所有靠 `ctx.emit` 做硬转移/
  意图路由的 agent(如 s-011 的 classify_intent/collect_slot/confirm_action)都无法完整 replay。

手法沿用 #73:把 emit 这个 handler 副作用**事件化**,replay 时按记录回灌,而非重跑 handler。

## 范围与非目标

**目标**:经 `executeSingleTool`(`ioPort.invokeTool`)的工具调用里 `ctx.emit` 触发的 FSM 转移,在
`replay()` 时确定性复现。

**非目标**:
- **action-state 的 emit**:`runActionState` 里 handler 是 `tool.handler(...)` **直跑**、不经
  `ioPort.invokeTool`,replay 时 handler 真跑、emit 自然触发,无需处理。
- **`ctx.requestSkill` 副作用**:handler 内请求 skill 在 replay 时同样会丢(同根因),会导致后续 LLM
  上下文缺 skill region → llm 散度。这是**相邻的同类 gap**,但不在 #60。建议作为 follow-up(#60 之于
  #73 的关系),后续单独开票。本次不扩入(避免过度工程)。

## 设计

### 1. 新事件类型 `tool.emitted`

`src/trace/types.ts`:`EventKind` 加 `'tool.emitted'`;新增 payload + alias 并入 `AnyEvent`。

```ts
export interface ToolEmittedPayload {
  /** 发出该 emit 的工具调用 id(LLM 指派)。来自 cached LLM 输出 → replay 时与录制一致,故可作对齐 key。 */
  toolCallId: string
  /** handler 经 ctx.emit 发出、且赢得 FSM 单一 pendingEvent 槽的那个业务事件。 */
  event: { name: string; payload?: unknown; guard?: GuardEvaluation[] }
}
```

不进 `CacheIndex`(与 `wm.mutated` 一样不是 nondet IO),`CacheIndex.fromEvents` 与 strict
under-consume 的四队列检查(clock/uuid/llm/tool)都自动忽略它。

### 2. 记录侧:只记"赢得 pendingEvent"的那次 emit

`AgentRuntime.executeSingleTool` 用**捕获式 emitFn** 判定本次调用是否真正赢得 FSM 的单一
`pendingEvent` 槽(`emitEvent` 是 first-wins):

```ts
let capturedEmit: { name: string; payload?: unknown; guard?: GuardEvaluation[] } | undefined
const ctx = this.buildToolContext((event, payload, guard) => {
  const hadPending = this.fsm.hasPendingEvent()
  this.fsm.emitEvent(event, payload, guard)
  if (!hadPending && this.fsm.hasPendingEvent()) {
    capturedEmit = { name: event, ...(payload !== undefined ? { payload } : {}), ...(guard ? { guard } : {}) }
  }
})
```

拿到 output 后(紧邻现有 `wm.mutated` 记录处),record 模式(`!this.replayEmits && this.eventStore`)
下若 `capturedEmit` 非空,`append` 一条 `tool.emitted { toolCallId: call.id, event: capturedEmit }`。

**为什么只记赢家**:并行 batch 里多个工具都可能 `ctx.emit`,但 first-wins 只有一个进 `pendingEvent`。
JS 单线程下 `emitEvent` 同步执行,`hadPending` 前后判定可靠地识别赢家。只记赢家 → replay 注入唯一一条
→ 转移与并发顺序无关地复现,消除并行竞态这一 record 层不确定性隐患。

### 3. 回放侧:按 `toolCallId` 注入

`Milkie.replay` 装配(与 `replayWmSnapshots` 平行):

```ts
replayEmits: new Map(events
  .filter(e => e.type === 'tool.emitted')
  .map(e => {
    const p = e.payload as ToolEmittedPayload
    return [p.toolCallId, p.event] as const
  }))
```

`AgentRuntime` 新增 opt `replayEmits?: Map<string, { name: string; payload?: unknown; guard?: GuardEvaluation[] }>`。
`executeSingleTool` replay 模式(`this.replayEmits` 存在)下,拿到 cached output 后:

```ts
const emitted = this.replayEmits.get(call.id)
if (emitted) this.fsm.emitEvent(emitted.name, emitted.payload, emitted.guard)
```

时序正确:`executeSingleTool` 在 `runLLMState` 内,注入发生在 `processPendingEvent()` 之前 →
`pendingEvent` 被设上 → 正常转移 → 不再多发 LLM → 不散度。replay 不传 `eventStore`,
`setupFSMCallbacks` 的 `fsm.transition` 写入被 `if (this.eventStore)` 跳过,符合"replay 不写事件"。

**对齐用 `toolCallId` 而非 positional**:`call.id` 源自 cached LLM 输出,replay 必然一致 → 稳;且
emit 稀疏,用 map 比"每 call 录一条(含空)"更省日志、无噪音。与 `wm.mutated` 的 positional shift 不同
是因为 WM 每 call 必变、emit 稀疏,各取所需。

## 测试(TDD,先红后绿)

1. **核心(升级现有绕过测试)**:`Replay.test.ts:419` 那个 `classify_intent`/`INTENT_DONE` emit-FSM,
   录一次后 `await milkie.replay(runId)` → 实现前抛 `ReplayDivergenceError(llm)`(红)→ 实现后
   resolves、output/最终状态一致、四队列消费干净(绿)。保留原 #31 断言不动。
2. **guard 透传**:emit 带 `guard` 的场景,replay 后转移仍发生,且注入路径不丢 guard 入参。
3. **不回归 #73**:不 emit 的普通 toolcall run replay 仍 OK。
4. **并行 first-wins 边界**:batch 内两个工具都 emit,replay 复现录制时那条生效转移。

## 改动文件

- `src/trace/types.ts` — 事件类型/payload/alias
- `src/runtime/AgentRuntime.ts` — 捕获式 emitFn + 记录 `tool.emitted`;`replayEmits` opt + 注入
- `src/runtime/Milkie.ts` — `replay()` 装配 `replayEmits`
- `src/__tests__/Replay.test.ts` — 新增真 replay 断言
- 文档:事件清单 + roadmap 标注 #60 落地、`requestSkill` follow-up

## Alternatives(已排除)

- **搭车 `tool.responded.emitted`**:`tool.responded` 由 `RecordingIOPort` 写,它不持有 FSM 句柄、
  拿不到 `pendingEvent`,记录侧塞不进(分层硬伤)。故新增独立事件。
- **复用 `fsm.transition`**:最省存储,但 business 转移与 tool call 的对齐需绕 causedBy,较脆。
- **方向 A(replay 重跑 handler)**:改 IOPort 语义、确定性边界复杂,且与 #73 既定模式相悖。
- **方向 C(文档化不支持)**:阻塞 Phase 5(fork/diff/suite 依赖 emit-FSM 可 replay)。
