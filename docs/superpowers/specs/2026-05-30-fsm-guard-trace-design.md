# #31 FSM guard 评估留痕 — 设计 spec

**Issue:** #31（依赖 #21 fsm.transition；阻塞 #33 why-this-transition；场景 s-011）
**日期:** 2026-05-30
**定位:** 小型可观测性补丁(选项 A 自觉上报),**不是**决策完备性的结构保证。

---

## 1. 问题与范围

milkie 的转移是纯事件名查表(`current.on[event.name]`),**没有框架级 guard**。"该跳哪"的判断溶解在应用工具/LLM 内部(如 s-011 `classify_intent` 里的 `confidence < 0.75`)。今天 trace 看得到"跳到了 handle_b",看不到"凭什么判成 B"。

本 issue 让工具在触发转移时,**可选地**把"判断依据"上报一句,写进 `fsm.transition`,使 routing 决策可在事后排查。

**采用选项 A(内嵌)**:依据挂在 `fsm.transition.payload` 的新字段上,不新增事件类型。

## 2. Non-goals(明确不做,避免被误读)

- **不建 guard 引擎**:框架不新增任何谓词求值;判断逻辑仍 100% 在应用工具内,本 issue 一行不改它。
- **不强制完备**:上报是 opt-in。工具不报 → 该转移无 guard 数据,**不报错**。trace 完备性在这一层依赖工具作者自觉——这是选项 A 的已知边界,不在本 issue 解决(若要结构化完备,需走"真 conditional guard",见 ARCHITECTURE.md Transition 概念的 planned extension)。
- **不强制 contextSlice 最小化**:框架无法知道"哪个字段决定了真假",只有工具自己知道。最小化是**约定**,不是校验。

## 3. 现状:主干其实已被结构化记录(决定本 issue 增量很小)

无需任何自觉,今天的 trace 已含:

| 已有事件 | 内容 | 来源 |
|---|---|---|
| `tool.requested` | 工具入参,如 `{intent:'cancellation', confidence:0.95}` | 框架自动(工具调用过 IOPort) |
| `fsm.transition.trigger.name` | `'INTENT_CANCELLATION'` | 框架自动 |
| `causedBy`(#30) | 转移 → 触发它的 `tool.responded` | 框架自动 |

即:LLM 的**感知**(intent/confidence)和**结论事件名**已可沿 `causedBy` 还原。本 issue 只补框架够不着的**增量**:
1. 工具内部**常量/阈值**(如 `threshold: 0.75`,不是入参,藏在代码里);
2. **哪些字段决定了真假**(最小化语义)。

## 4. 数据结构

`src/trace/types.ts`:

```ts
export interface GuardEvaluation {
  /** 判断标识,如 'intent-threshold'。 */
  guardId:      string
  /** 判断结果:产出的事件名或布尔/任意值,如 'INTENT_CANCELLATION'。 */
  result:       unknown
  /** ★ 决定结果真假的最小输入切片,如 { intent, confidence, threshold }。约定最小化。 */
  contextSlice: unknown
}

export interface FsmTransitionPayload {
  from: string
  to:   string
  trigger: { domain: FsmEventDomain; name: string; payload?: unknown }   // 不变
  /** #31:本次转移背后的判断依据(工具自报,可选)。 */
  guardEvaluations?: GuardEvaluation[]
}
```

`guardEvaluations` 与 `trigger.payload` **严格分离**——后者是"事件随带数据"的杂物位,前者语义明确为"为什么这么判"。两者不复用同一字段。

## 5. 上报 API(authoring)

**选定:给 `ctx.emit` 增加独立的第三参 `guard`**(`src/types/tool.ts`):

```ts
emit: (event: string, payload?: unknown, guard?: GuardEvaluation | GuardEvaluation[]) => void
```

工具改动(s-011 `classify_intent`,一处):

```ts
// 今天
ctx.emit(eventMap[intent] ?? 'ESCALATE')

// #31 之后
ctx.emit(eventMap[intent] ?? 'ESCALATE', undefined, {
  guardId:      'intent-threshold',
  result:       eventMap[intent] ?? 'ESCALATE',
  contextSlice: { intent, confidence, threshold: 0.75 },
})
```

理由:① 现状 `emit` 第二参 `payload` 实际无人使用,但保留以免破坏签名;② guard 作为独立第三参,与 `trigger.payload` 语义不混;③ 一次调用原子完成"发事件 + 报依据",guard 恰好对应产生本次转移的那一次决策,无需缓冲生命周期。

**决定:采用甲,锁定。** 备选乙(单独 `ctx.recordGuard(evaluation)` 方法,多次调用累积成数组、由下一次转移消费)已评估**不采用**——它引入"记录了却没转移""记录顺序""跨 emit 归属"等缓冲生命周期边界,违背最小自洽;甲 的一次调用原子绑定"发事件 + 报依据",更简单。

## 6. 捕获机制(复用 #30 模式,零新增 IOPort 调用)

1. `FSMEngine.emitEvent(event, payload?, guard?)`:第三参存入 `pendingEvent`(`{ name, payload, domain, guard }`);"first event wins" 逻辑不变。
2. `FSMEvent` 类型(`FSMEngine.ts`)加可选 `guard?: GuardEvaluation[]`(单个则归一化为数组)。
3. `transition()` 的 `onTransition(from, to, event)` 回调已带 `event`,guard 随之传出。
4. `AgentRuntime.setupFSMCallbacks`(`:170-211`)组装 `fsm.transition` payload 时,写入 `...(event.guard?.length ? { guardEvaluations: event.guard } : {})`。仍用 `uuidv4()/Date.now()` 直接生成 id/timestamp,**绕过 IOPort**。
5. 框架自身的 lifecycle/signal/runtime-control 事件(DONE/interrupt/error/RETRY)不带 guard,自然无此字段。

## 7. Replay / byte-identical

- guardEvaluations 仅是 `fsm.transition` 上的额外 payload 内容。`fsm.transition` 本就绕过 IOPort、不进 nondet 缓存(`AgentRuntime.ts:178-183`)。
- 因此本改动**不产生任何新的 `ioPort.uuid()/now()` 调用**,回放消费的 LLM/tool/clock/uuid 缓存序列与改动前**逐字节一致** → replay 确定性不破。这是 #31 "replay byte-identical" 验收的精确含义。
- guard 数据在**录制时**(handler 真跑、`ctx.emit` 触发)捕获;回放时 handler 不重跑(`ReplayingIOPort.invokeTool` 忽略 execute thunk),这是既有行为,本 issue 不改变、也不依赖它。诊断对象是录制 run,数据齐全。
- 诚实声明:`fsm.transition` 事件的 `id/timestamp` 在录制/回放间本就不同(直接生成),事件**日志字节**对这些信息性事件本就非逐字节一致——这是既有事实,非本 issue 引入。

## 8. Producer 接线(给本 issue 一个真实水源)

改 s-011 三个工具,各在 emit 处补一次上报,使端到端 trace 真能看到 guard:

| 工具 | guardId | contextSlice |
|---|---|---|
| `classify_intent` | `intent-threshold` | `{ intent, confidence, threshold: 0.75 }` |
| `collect_slot` | `slots-complete` | `{ filled: [...], required: ['orderId','reason','preferRefund'] }` |
| `confirm_action` | `user-confirm` | `{ confirmed }` |

## 9. 渲染

`src/trace/render/`:`fsm.transition` 条目下,若有 `guardEvaluations`,多渲一行,如:
`guard intent-threshold → INTENT_CANCELLATION（intent=cancellation, confidence=0.95 ≥ 0.75）`。无该字段时行为不变。

## 10. 测试

- **单元**:一个工具 `ctx.emit(name, undefined, guard)` → 断言对应 `fsm.transition` 事件 payload 含 `guardEvaluations`,内容匹配。
- **不带 guard 的转移**:断言 payload **无** `guardEvaluations` 字段(保持干净、向后兼容)。
- **replay 确定性**:沿用 s-005 模式——带 guard 的 run 回放后 `status/output` 一致、gateway 调用数为 0,证明缓存序列未受扰。
- **渲染**:带 guard 的事件渲出额外行;不带的不渲。
- **s-011 e2e**:live 路径下,trace 中 `classify_intent` 触发的 `fsm.transition` 含 `intent-threshold` 的 guardEvaluations。

> **验证分工(诚实声明)**:s-011 是 live e2e,挂在 `VOLCENGINE_TOKEN` 后,无 token 即 skip——故 **每次都跑的确定性验证是单元测试**(自带假工具 + mock gateway);s-011 提供"真实多状态场景也通"的额外信心与一份 authoring 样例,但常在 CI 被 skip。两者都要:单元测试保确定性,s-011 保真实性。

## 11. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/types.ts` | +`GuardEvaluation`;`FsmTransitionPayload` 加 `guardEvaluations?` |
| `src/types/tool.ts` | `emit` 签名加第三参 `guard?` |
| `src/fsm/FSMEngine.ts` | `FSMEvent` 加 `guard?`;`emitEvent` 加第三参并存入 `pendingEvent` |
| `src/runtime/AgentRuntime.ts` | `buildToolContext` 透传 guard;`setupFSMCallbacks` 写入 `guardEvaluations` |
| `src/trace/render/*` | 渲染分支 |
| `tests/e2e/s-011-*.ts` | 三个工具补上报 |
| `src/__tests__/*` | 新增单元 + replay 测试 |

## 12. 验收对照(#31)

- [x] 多 guard FSM 的事件流可还原"哪些 guard 真求值、值多少" → guardEvaluations(录制时捕获)
- [x] contextSlice 最小化 → 约定 + s-011 示例;**不强制**(已在 Non-goals 声明)
- [x] replay byte-identical → 零新增 IOPort 调用,缓存序列不变
