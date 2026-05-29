# sub-agent 一类公民化（模型 I）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让子 agent 把自己的 I/O 录进独立的 `<childRunId>.jsonl`、emit 自己的 `agent.run.started/completed`，并让含 sub-agent 的 run 能正确 replay —— 不引入递归 replay（模型 I）。

**Architecture:** Milkie 注入一个 record-only 的子 port 工厂；spawn 时给子铸一个绑 `childRunId`、用子自己 model gateway 的 `RecordingIOPort`，子 I/O 与生命周期事件落子的流。父 replay **不重跑子**（子 tool 由父 cache 吐 output）；为此把父侧子边界 id 改为信息性 uuid，使父 cache 不含任何"只有重跑子才会消费"的条目。子可经 `Milkie.replay(childRunId)` 独立重放。

**Tech Stack:** TypeScript (ESM, `.js` import 后缀)、vitest。事件存储 `IEventStore`（Memory/Jsonl），IOPort 装饰链（`DefaultIOPort` → `RecordingIOPort`/`ReplayingIOPort`）。

**设计依据：** `docs/superpowers/specs/2026-05-29-sub-agent-first-class-replay-design.md`

---

## File Structure

- `src/types/store.ts` — `ChildAgentRecord` 加 `runId`。
- `src/runtime/AgentFactory.ts` — `AgentSpawnOptions` 加 `eventStore` + `makeChildPort`（向下传递子构造所需）。
- `src/runtime/Milkie.ts` — 构造 `makeChildPort` 工厂并注入顶层 runtime（`invoke` / `resume`）。
- `src/runtime/AgentRuntime.ts` — 主体：`makeChildPort` option/field/构造；`spawnFn` 注入向下传递；`makeSubAgentTool.handler` 改写（独立 childRunId、子 port、生命周期、信息性父侧 id）。
- `src/__tests__/Trace.test.ts` — record 侧断言（子流分离、生命周期、父流干净、gateway）。
- `src/__tests__/Replay.test.ts` — 父 replay 转绿 + 子独立 replay。

每个 Task 自带测试，先红后绿，独立提交。

---

## Task 1: 子获得独立 runId，I/O 落 `<childRunId>.jsonl`

**Files:**
- Modify: `src/runtime/AgentFactory.ts:8-19`（`AgentSpawnOptions`）
- Modify: `src/runtime/AgentRuntime.ts:40-62`（options）、`75-118`（field+ctor）、`135-142`（spawnFn）、`210-247`（handler）
- Modify: `src/runtime/Milkie.ts:197-213`（invoke 注入）
- Test: `src/__tests__/Trace.test.ts`

- [ ] **Step 1: 写失败测试** — 在 `Trace.test.ts` 的 `describe('agent.spawned / agent.returned events', ...)` 内追加：

```ts
it('records child LLM I/O under an independent childRunId, not the parent run', async () => {
  const eventStore = new MemoryEventStore()
  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    gateway: new StubGateway([
      toolCallResponse('s1', 'worker', { goal: 'subgoal', input: 'subinput' }),
      textResponse('worker done'),   // worker 的 LLM 回复
      textResponse('all done'),      // supervisor 收尾
    ]),
    eventStore,
  })
  milkie.registerAgent(supervisorConfig())
  milkie.registerAgent(workerConfig())

  const result = await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })
  const parentEvents = await eventStore.readByRunId(result.agentRunId)

  const spawned = parentEvents.find(e => e.type === 'agent.spawned')!.payload as AgentSpawnedPayload
  // childRunId 现在是真正独立的 runId，且 != 父 runId
  expect(spawned.childRunId).not.toBe(result.agentRunId)

  // 父流里没有 worker 的 llm 事件（它们去了子流）
  const parentLlm = parentEvents.filter(e => e.type === 'llm.requested')
  // supervisor 自己有 2 次 LLM（spawn 决策 + 收尾），worker 的不在父流
  const childEvents = await eventStore.readByRunId(spawned.childRunId)
  const childLlm = childEvents.filter(e => e.type === 'llm.requested')
  expect(childLlm.length).toBeGreaterThan(0)        // 子流有 worker 的 llm
  expect(parentLlm.length).toBe(2)                  // 父流只有 supervisor 自己的
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx vitest run src/__tests__/Trace.test.ts -t "independent childRunId"`
Expected: FAIL（今天子复用父 runId/port，`spawned.childRunId` 是 contextId 且子 llm 落在父流，`childLlm.length` 为 0 或 `parentLlm` 偏多）

- [ ] **Step 3a: `AgentSpawnOptions` 加字段** — `src/runtime/AgentFactory.ts`，在 interface 内 `extraTools?` 后追加：

```ts
  eventStore?:   import('../trace/EventStore.js').IEventStore
  makeChildPort?: import('./AgentRuntime.js').MakeChildPort
```

- [ ] **Step 3b: 在 `AgentRuntime.ts` 定义并导出 `MakeChildPort` 类型** — 紧邻 `AgentRuntimeOptions`（约 `:39` 之后）：

```ts
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../trace/types.js'

export type MakeChildPort = (
  childRunId:  string,
  childConfig: AgentConfig,
  start:       AgentRunStartedPayload,
) => Promise<{ port: IIOPort; finish: (c: AgentRunCompletedPayload) => Promise<void> }>
```

（若文件已 `import type { SkillLifecyclePayload } from '../trace/types.js'`，把上面两个类型并入同一 import。）

- [ ] **Step 3c: options + field + 构造 + spawnFn 向下传递** — `AgentRuntime.ts`：

`AgentRuntimeOptions` 内 `childRecorderFactory?` 后加：
```ts
  makeChildPort?: MakeChildPort
```
field 区（`childRecorderFactory` 私有字段后）加：
```ts
  private readonly makeChildPort?: MakeChildPort
```
构造函数体（`this.childRecorderFactory = opts.childRecorderFactory` 附近）加：
```ts
    this.makeChildPort = opts.makeChildPort
```
spawnFn（`:135-142`）改为把 `eventStore` / `makeChildPort` 也传给子，使孙子也能分流：
```ts
    this.factory = new AgentFactory(async (spawnOpts: AgentSpawnOptions) => {
      const child = new AgentRuntime({
        ...spawnOpts,
        parentId:             this.agentRunId,
        childRecorderFactory: this.childRecorderFactory,
        eventStore:           this.eventStore,
        makeChildPort:        this.makeChildPort,
      })
      return child.run(spawnOpts.input)
    })
```

- [ ] **Step 3d: 改写 `makeSubAgentTool.handler`** — `AgentRuntime.ts:210-280`。关键改动：父侧子边界 id 用 `uuidv4()`（信息性，绕 cache）；新增独立 `childRunId`；铸子 port + 生命周期；spawn 用子 port + 独立 runId；`agent.spawned/returned` 用 `childRunId`。替换 handler 体内自 `const childTraceId` 起到 spawn 调用：

```ts
      handler: async (rawInput: unknown, ctx: ToolContext) => {
        const { goal, input: subInput } = rawInput as { goal: string; input: string }
        const subConfig = this.subAgentConfigs?.get(agentId)
        if (!subConfig) {
          throw new Error(`Sub-agent config not found: ${agentId}. Pass subAgentConfigs to AgentRuntime.`)
        }

        // 父侧子边界 id 信息性化（绕 cache）：模型 I 下父 replay 不进 handler，
        // 这些不可是父 cache 条目，否则 tail under-consume。
        const childTraceId   = uuidv4()
        const childContextId = uuidv4()
        const taskId         = uuidv4()
        const childRunId     = uuidv4()   // 子的独立 run 身份

        const childRecorder = this.childRecorderFactory?.(subConfig, childContextId, childTraceId) ?? this.recorder
        const spawnSpan = this.recorder.startSpan('agent.spawn', {
          childAgentId: agentId, taskId, turn: this.turnNumber, childTraceId, childContextId, childRunId,
        })
        await this.recordChild({ taskId, agentId, runId: childRunId, contextId: childContextId, status: 'running' })
        this.emitAgentSpawned(childRunId, agentId, goal)

        // 铸子 port（record-only；replay 不注入 makeChildPort → 回退父 ioPort，且 replay 不进此分支）
        let childPort: IIOPort = this.ioPort
        let finish: ((c: AgentRunCompletedPayload) => Promise<void>) | null = null
        if (this.makeChildPort) {
          const built = await this.makeChildPort(childRunId, subConfig, {
            agentId, goal, input: subInput, contextId: childContextId, parentId: this.agentRunId,
          })
          childPort = built.port
          finish    = built.finish
        }

        try {
          const result = await ctx.agentFactory.spawn({
            config:     subConfig,
            goal,
            input:      subInput,
            contextId:  childContextId,
            agentRunId: childRunId,        // 独立 runId
            stateStore: this.stateStore,
            recorder:   childRecorder,
            ioPort:     childPort,         // 子自己的 port
            extraTools: this.extraTools,
            eventStore: this.eventStore,   // 子的 fsm/region 也落子流
            makeChildPort: this.makeChildPort,
          })
          await finish?.({ status: result.status, lastTextOutput: result.output })

          const childCheckpoint = result.status === 'interrupted'
            ? await this.stateStore.get(`context:${childContextId}:checkpoint:latest`) as AgentCheckpoint | undefined
            : undefined
          await this.recordChild({
            taskId, agentId, runId: childRunId, contextId: childContextId,
            checkpointId: childCheckpoint?.checkpointId,
            status: result.status === 'interrupted' ? 'interrupted' : result.status === 'completed' ? 'success' : 'error',
          })
          this.recorder.recordEvent(spawnSpan, 'agent.spawn.complete', { resultStatus: result.status })
          this.emitAgentReturned(childRunId, result.status)
          spawnSpan.attributes['resultStatus']   = result.status
          spawnSpan.attributes['childTraceId']   = childTraceId
          spawnSpan.attributes['childContextId'] = childContextId
          if (childCheckpoint?.checkpointId) spawnSpan.attributes['checkpointId'] = childCheckpoint.checkpointId
          this.recorder.endSpan(spawnSpan, 'ok')
          return result.output
        } catch (err) {
          await finish?.({ status: 'error', error: err instanceof Error ? err.message : String(err) })
          await this.recordChild({ taskId, agentId, runId: childRunId, contextId: childContextId, status: 'error' })
          this.emitAgentReturned(childRunId, 'error')
          spawnSpan.attributes['resultStatus'] = 'error'
          this.recorder.endSpan(spawnSpan, 'error')
          throw err
        }
      },
```

确保文件顶部已 import `AgentRunCompletedPayload`（Step 3b 已加）。

- [ ] **Step 3e: `ChildAgentRecord` 加 `runId`** — `src/types/store.ts:48-54`：

```ts
export interface ChildAgentRecord {
  taskId:        string
  agentId:       string
  runId?:        string
  contextId?:    string
  checkpointId?: string
  status:        'running' | 'success' | 'error' | 'interrupted'
}
```

- [ ] **Step 3f: Milkie 构造并注入 `makeChildPort`** — `src/runtime/Milkie.ts`。在 `invoke`（`:197` 附近，构造 runtime 前）加：

```ts
    const makeChildPort = this.eventStore
      ? (async (childRunId, childConfig, start) => {
          const gw   = this.gatewayOverride ?? createGateway(childConfig.model)
          const port = new RecordingIOPort(new DefaultIOPort(gw), this.eventStore!, childRunId)
          await port.attach(start)
          return { port, finish: (c) => port.detach(c) }
        }) satisfies import('./AgentRuntime.js').MakeChildPort
      : undefined
```
并在 `new AgentRuntime({...})` 的 opts 里加 `makeChildPort,`。`resume`（`:276`）同样加 `makeChildPort`（可抽一个私有 `buildMakeChildPort()` 复用，避免重复）。

- [ ] **Step 4: 跑测试确认通过**

Run: `npx vitest run src/__tests__/Trace.test.ts -t "independent childRunId"`
Expected: PASS

- [ ] **Step 5: 跑既有 #24 spawn 测试确认未回归**

Run: `npx vitest run src/__tests__/Trace.test.ts -t "agent.spawned"`
Expected: PASS（`spawned.childRunId === returned.childRunId` 仍成立，值变为真 childRunId）

- [ ] **Step 6: Commit**

```bash
git add src/types/store.ts src/runtime/AgentFactory.ts src/runtime/AgentRuntime.ts src/runtime/Milkie.ts src/__tests__/Trace.test.ts
git commit -m "feat(trace): sub-agent records I/O under independent childRunId (#47)"
```

---

## Task 2: 子 emit `agent.run.started{parentId}` / `agent.run.completed{status}`

子的生命周期事件已由 Task 1 的 `makeChildPort`（`attach`/`finish→detach`）写入子流。本 Task 加断言锁定契约。

**Files:**
- Test: `src/__tests__/Trace.test.ts`

- [ ] **Step 1: 写测试**

```ts
it('child run emits agent.run.started{parentId} and agent.run.completed in its own stream', async () => {
  const eventStore = new MemoryEventStore()
  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    gateway: new StubGateway([
      toolCallResponse('s1', 'worker', { goal: 'subgoal', input: 'subinput' }),
      textResponse('worker done'),
      textResponse('all done'),
    ]),
    eventStore,
  })
  milkie.registerAgent(supervisorConfig())
  milkie.registerAgent(workerConfig())

  const result = await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })
  const parentEvents = await eventStore.readByRunId(result.agentRunId)
  const spawned = parentEvents.find(e => e.type === 'agent.spawned')!.payload as AgentSpawnedPayload

  const childEvents = await eventStore.readByRunId(spawned.childRunId)
  const started = childEvents.find(e => e.type === 'agent.run.started')!
  const completed = childEvents.find(e => e.type === 'agent.run.completed')!
  expect(childEvents[0]!.type).toBe('agent.run.started')         // 子流第一帧
  expect((started.payload as AgentRunStartedPayload).parentId).toBe(result.agentRunId)
  expect((started.payload as AgentRunStartedPayload).agentId).toBe('worker')
  expect((completed.payload as AgentRunCompletedPayload).status).toBe('completed')
  expect(started.runId).toBe(spawned.childRunId)
})
```

（在 `Trace.test.ts` 顶部 import 补 `AgentRunStartedPayload, AgentRunCompletedPayload`。）

- [ ] **Step 2: 跑测试**

Run: `npx vitest run src/__tests__/Trace.test.ts -t "agent.run.started{parentId}"`
Expected: PASS（Task 1 已实现行为；若失败说明 attach 顺序/parentId 未填，回到 Task 1 Step 3f 检查）

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/Trace.test.ts
git commit -m "test(trace): assert child emits run.started{parentId}/completed (#47)"
```

---

## Task 3: 含 sub-agent 的父 run 能正确 replay（核心验收，今天是坏的）

**Files:**
- Test: `src/__tests__/Replay.test.ts`

- [ ] **Step 1: 写失败测试** — 仿 `Replay.test.ts` 既有 record→replay 双 Milkie 模式：

```ts
it('replays a run containing a sub-agent with no divergence', async () => {
  const eventStore = new MemoryEventStore()
  const record = new Milkie({
    stateStore: new MemoryStore(),
    gateway: new StubGateway([
      toolCallResponse('s1', 'worker', { goal: 'subgoal', input: 'subinput' }),
      textResponse('worker done'),
      textResponse('all done'),
    ]),
    eventStore,
  })
  record.registerAgent(supervisorConfig())
  record.registerAgent(workerConfig())
  const orig = await record.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })

  const replay = new Milkie({
    stateStore: new MemoryStore(),
    gateway: new StubGateway([]),   // replay 不应触达 gateway
    eventStore,
  })
  replay.registerAgent(supervisorConfig())
  replay.registerAgent(workerConfig())

  const replayed = await replay.replay(orig.agentRunId)
  expect(replayed.status).toBe('completed')
  expect(replayed.output).toBe(orig.output)   // 'all done'
})
```

（`supervisorConfig`/`workerConfig`/`StubGateway`/`toolCallResponse`/`textResponse` 若 `Replay.test.ts` 没有，从 `Trace.test.ts` 复制这几个 helper 到本文件顶部。）

- [ ] **Step 2: 跑测试确认失败（前提：未做 §1d 之前）**

Run: `npx vitest run src/__tests__/Replay.test.ts -t "containing a sub-agent"`
注意：若 Task 1 已落地（父侧 id 已信息性化、子 I/O 已分流），此测试**可能直接 PASS** —— 这正是模型 I 的红利（replay 侧零改动）。若 PASS，记录"今天坏、Task1 后转绿"的事实于 commit message，跳到 Step 4。若仍 FAIL（divergence），按 Step 3 排查。

- [ ] **Step 3: 排查（仅当 Step 2 FAIL）**

divergence 必为 over/under-consume 之一：
- **under-consume**：父 cache 仍含"只有重跑子才消费"的条目 → 检查是否还有父侧子边界值走了 `this.ioPort.uuid()`/`this.ioPort.now()`（应全为 `uuidv4()`/recorder）。`batchId`（`AgentRuntime.ts:1009`）保持 ioPort.uuid 是对的（handler 外、record/replay 对称消费）。
- **over-consume**：子 tool 的 `tool.responded` 未被父 cache 命中 → 检查 `hashToolCall(agentId, {goal,input})` 在 record/replay 是否一致（input 形状一致即可）。

修正后回到 Step 2。

- [ ] **Step 4: 跑测试确认通过**

Run: `npx vitest run src/__tests__/Replay.test.ts -t "containing a sub-agent"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/__tests__/Replay.test.ts
git commit -m "test(trace): parent run with sub-agent replays without divergence (#47)"
```

---

## Task 4: 子 run 可经 `Milkie.replay(childRunId)` 独立重放

**Files:**
- Test: `src/__tests__/Replay.test.ts`

- [ ] **Step 1: 写测试**

```ts
it('replays a child run standalone by its childRunId', async () => {
  const eventStore = new MemoryEventStore()
  const record = new Milkie({
    stateStore: new MemoryStore(),
    gateway: new StubGateway([
      toolCallResponse('s1', 'worker', { goal: 'subgoal', input: 'subinput' }),
      textResponse('worker done'),
      textResponse('all done'),
    ]),
    eventStore,
  })
  record.registerAgent(supervisorConfig())
  record.registerAgent(workerConfig())
  const orig = await record.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })
  const parentEvents = await eventStore.readByRunId(orig.agentRunId)
  const childRunId = (parentEvents.find(e => e.type === 'agent.spawned')!.payload as AgentSpawnedPayload).childRunId

  const replay = new Milkie({ stateStore: new MemoryStore(), gateway: new StubGateway([]), eventStore })
  replay.registerAgent(supervisorConfig())
  replay.registerAgent(workerConfig())

  const child = await replay.replay(childRunId)
  expect(child.status).toBe('completed')
  expect(child.output).toBe('worker done')
})
```

- [ ] **Step 2: 跑测试**

Run: `npx vitest run src/__tests__/Replay.test.ts -t "child run standalone"`
Expected: PASS（子流自带 `agent.run.started`(goal/input/contextId/parentId) + llm I/O → `extractRunSnapshot` + `CacheIndex` 即可独立重放）。若 FAIL 报 "missing lifecycle event"，确认 Task 1 的子 `attach/detach` 已写入子流。

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/Replay.test.ts
git commit -m "test(trace): child run is independently replayable by childRunId (#47)"
```

---

## Task 5: `ChildAgentRecord.runId` 持久化进父 checkpoint

**Files:**
- Test: `src/__tests__/AgentRuntime.test.ts`

- [ ] **Step 1: 写测试** — 仿既有 supervisor 中断测试（`AgentRuntime.test.ts:375` 一带），断言父 checkpoint 的 `children[].runId` 等于 `agent.spawned.childRunId`。复用其 harness（interrupt → 读 `context:<ctx>:checkpoint:latest`）：

```ts
it('persists child runId in the parent checkpoint children records', async () => {
  // ...沿用 :375 测试的 supervisor+worker+eventStore+interrupt 搭建...
  // interrupt 后：
  const parentCp = await stateStore.get(`context:ctx-supervisor:checkpoint:latest`) as AgentCheckpoint
  expect(parentCp.children.length).toBeGreaterThan(0)
  for (const c of parentCp.children) {
    expect(typeof c.runId).toBe('string')
    expect(c.runId!.length).toBeGreaterThan(0)
  }
})
```

- [ ] **Step 2: 跑测试**

Run: `npx vitest run src/__tests__/AgentRuntime.test.ts -t "persists child runId"`
Expected: PASS（Task 1 Step 3d 的 `recordChild({ runId: childRunId, ... })` + checkpoint 已存 `children`，`AgentRuntime.ts:717`）。若 FAIL，确认 `recordChild` 各调用点都带 `runId`。

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/AgentRuntime.test.ts
git commit -m "test(trace): persist child runId in parent checkpoint (#47)"
```

---

## Task 6: 子用自己 model 的 gateway（修连带 bug）

**Files:**
- Test: `src/__tests__/Trace.test.ts`

- [ ] **Step 1: 写测试** — 不设 `gateway` override；让子 config 的 adapter 非法，spawn 时 `createGateway(childConfig.model)` 会抛、且错误信息含子的 adapter 名 → 证明子 port 用的是 **childConfig.model**（而非父 gateway）：

```ts
it('builds the child port from the child config gateway (not the parent)', async () => {
  const eventStore = new MemoryEventStore()
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore })   // 无 gateway override

  const parent = supervisorConfig()
  parent.model = { provider: 'anthropic', model: 'm', adapter: 'anthropic' } // 父：可构造、无网络调用
  const child = workerConfig()
  child.model = { provider: 'x', model: 'm', adapter: 'no-such-adapter' }     // 子：非法 adapter
  milkie.registerAgent(parent)
  milkie.registerAgent(child)

  const result = await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })
  const events = await eventStore.readByRunId(result.agentRunId)
  const toolResp = events.find(e => e.type === 'tool.responded' &&
    (e.payload as ToolRespondedPayload).toolName === 'worker')!.payload as ToolRespondedPayload
  expect(toolResp.error?.message ?? '').toContain('no-such-adapter')
})
```

注意：父 LLM 需先返回一个调用 worker 的 tool_use。无 override 时父 `anthropic` gateway 会真的发起调用 → 测试不可控。**因此本测试改用 record 路径不可行**；替代为**直接单测 `makeChildPort` 行为**：

```ts
it('child port factory resolves gateway from child config', async () => {
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
  const make = (milkie as any)['buildMakeChildPort']?.() ?? null
  expect(make).not.toBeNull()
  const bogus = { ...workerConfig(), model: { provider: 'x', model: 'm', adapter: 'no-such-adapter' } }
  await expect(make('child-run-1', bogus, {
    agentId: 'worker', goal: 'g', input: 'i', contextId: 'c', parentId: 'p',
  })).rejects.toThrow(/no-such-adapter/)
})
```

为此把 Task 1 Step 3f 的工厂抽成 `private buildMakeChildPort(): MakeChildPort | undefined`（invoke/resume 复用），测试经 `as any` 取到它。**采用此单测版本，删除上面那个不可控的 record 版本。**

- [ ] **Step 2: 跑测试确认失败**

Run: `npx vitest run src/__tests__/Trace.test.ts -t "child config"`
Expected: FAIL（若 `buildMakeChildPort` 尚未抽出 → undefined；或工厂误用了 override/父 gateway）

- [ ] **Step 3: 实现** — 在 `Milkie.ts` 把工厂抽成方法：

```ts
private buildMakeChildPort(): import('./AgentRuntime.js').MakeChildPort | undefined {
  if (!this.eventStore) return undefined
  const eventStore = this.eventStore
  const override   = this.gatewayOverride
  return async (childRunId, childConfig, start) => {
    const gw   = override ?? createGateway(childConfig.model)
    const port = new RecordingIOPort(new DefaultIOPort(gw), eventStore, childRunId)
    await port.attach(start)
    return { port, finish: (c) => port.detach(c) }
  }
}
```
`invoke`/`resume` 改用 `const makeChildPort = this.buildMakeChildPort()`。

- [ ] **Step 4: 跑测试确认通过**

Run: `npx vitest run src/__tests__/Trace.test.ts -t "child config"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runtime/Milkie.ts src/__tests__/Trace.test.ts
git commit -m "feat(trace): child port uses child's own model gateway (#47)"
```

---

## Task 7: 全量回归

- [ ] **Step 1: 跑整个测试套件**

Run: `npx vitest run`
Expected: 全绿。重点确认无回归：`Replay.test.ts`（既有单 run）、`AgentRuntime.test.ts`（supervisor 中断/恢复、checkpoint）、`Trace.test.ts`（#24 spawn/returned、fsm.transition、skill 生命周期）。

- [ ] **Step 2: 类型检查 / 构建**

Run: `npx tsc --noEmit`（或项目既有 `npm run build` / `npm run typecheck`，先 `cat package.json` 确认脚本名）
Expected: 无类型错误。

- [ ] **Step 3: 若有失败** 按 superpowers:systematic-debugging 逐个定位；修复后回到 Step 1。不得用 `--no-verify` 跳过。

---

## Self-Review 备忘（计划作者已核对）

- **Spec 覆盖**：独立 runId+子流(Task1) / 子生命周期(Task2) / 父 replay 转绿(Task3) / 子独立 replay(Task4) / childRunId 持久化(Task5) / 子 gateway(Task6) / 回归(Task7)。`agent.spawned/returned.childRunId` 升级为真 runId 在 Task1 落地、Task1 Step5 验证 #24 不回归。
- **deferral**（spec 已列）：递归 replay（模型 II）、父 resume→子按稳定键续跑的**执行**、fork、`causedBy` —— 本计划不含，符合预期。
- **类型一致**：`MakeChildPort`（AgentRuntime 导出）= `(childRunId, childConfig, start) => Promise<{port, finish}>`，在 AgentFactory/Milkie/AgentRuntime 三处签名一致；`ChildAgentRecord.runId?: string` 在 store.ts 定义、recordChild 各调用点填充。
- **已知取舍**：`makeChildPort` 仅在 `eventStore` 存在时注入 ⟺ 子才有独立 RecordingIOPort；无 eventStore 时子 port 回退父 ioPort（DefaultIOPort，不录事件，无 runId 错配）。
