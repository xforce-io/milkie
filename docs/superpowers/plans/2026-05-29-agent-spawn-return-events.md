# agent.spawned / agent.returned 事件化（#24）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在父 agent run 的 event log 里 emit `agent.spawned` / `agent.returned` 锚点，关掉「supervisor tree 只能从 `ChildAgentRecord` 反推」这个 observable 洞。

**Architecture:** 子 agent 仍复用父 runId（独立 runId / sub-trace 是 #47 的后续工作）。父在 `makeSubAgentTool` 的 spawn/return 边界，经 `enqueueTraceWrite`（有序 + best-effort + LLM 前 flush）写两类新事件。`childRunId` 字段当前填 `childContextId`，#47 再升级为真 runId（只换值不换 schema）。附带把 `emitSkillLifecycle` 从 `void append` 迁到同一条 `enqueueTraceWrite` 路径。

**Tech Stack:** TypeScript / jest（`npx jest <file> --runInBand`）。

设计依据：`docs/superpowers/specs/2026-05-29-agent-spawn-return-events-design.md`

---

## File Structure

- `src/trace/types.ts` — 加 `EventKind` 两项、`AgentSpawnedPayload` / `AgentReturnedPayload`、两个 typed event 别名。
- `src/runtime/AgentRuntime.ts` — 加 `emitAgentSpawned` / `emitAgentReturned` 两个私有 helper；在 `makeSubAgentTool` 三处 emit；把 `emitSkillLifecycle` 的 `void append` 改成 `enqueueTraceWrite`。
- `src/__tests__/Trace.test.ts` — 新增 `agent.spawned / agent.returned` 的 describe（completed / error 两态 + 顺序）。
- `src/__tests__/AgentRuntime.test.ts` — 新增 interrupted 态测试（复用现有 interrupt harness + eventStore）。

---

### Task 1: 事件 taxonomy + payload（types.ts）

**Files:**
- Modify: `src/trace/types.ts:9-23`（EventKind）、`src/trace/types.ts:88-92`（payload 区）、`src/trace/types.ts:184-185`（event 别名区）

- [ ] **Step 1: 在 EventKind 末尾加两项**

`src/trace/types.ts`，把第 23 行 `  | 'skill.unloaded'` 之后补两行：

```ts
  | 'skill.unloaded'
  | 'agent.spawned'
  | 'agent.returned'
```

- [ ] **Step 2: 在 lifecycle payload 区加两个 payload**

紧接 `AgentRunCompletedPayload`（约 88-92 行）之后插入：

```ts
export interface AgentSpawnedPayload {
  /** 父 AgentRuntime.agentRunId。 */
  parentRunId: string
  /**
   * 子运行的稳定身份。今天子复用父 runId，此字段填子的 contextId；
   * #47（sub-agent 一类公民化）落地后改填子的独立 runId——只换值不换 schema。
   */
  childRunId:  string
  agentId:     string
  goal:        string
}

export interface AgentReturnedPayload {
  childRunId: string
  status:     'completed' | 'interrupted' | 'error'
}
```

- [ ] **Step 3: 加两个 typed event 别名**

紧接 `AgentRunCompletedEvent`（第 185 行）之后插入：

```ts
export type AgentSpawnedEvent  = Event<AgentSpawnedPayload>  & { type: 'agent.spawned' }
export type AgentReturnedEvent = Event<AgentReturnedPayload> & { type: 'agent.returned' }
```

- [ ] **Step 4: 构建验证类型编译**

Run: `npm run build`
Expected: tsc 无报错（仅类型新增，无消费者，应干净通过）。

- [ ] **Step 5: Commit**

```bash
git add src/trace/types.ts
git commit -m "feat(trace): add agent.spawned/returned event taxonomy (#24)"
```

---

### Task 2: emit helper + 接 spawn 成功路径 + completed 测试

**Files:**
- Test: `src/__tests__/Trace.test.ts`（新增 describe）
- Modify: `src/runtime/AgentRuntime.ts`（加 helper + `makeSubAgentTool` 两处 emit）

- [ ] **Step 1: 写失败测试（completed 态）**

`src/__tests__/Trace.test.ts`，先在文件顶部类型 import 块（约第 13-25 行 `from '../trace/types'`）补两个类型：

```ts
  RegionAddedPayload,
  SkillLifecyclePayload,
  AgentSpawnedPayload,
  AgentReturnedPayload,
```

然后在 `// ---- fsm.transition (#21) ----` 这一行（约第 540 行）之前插入：

```ts
// ---- agent.spawned / agent.returned (#24) ----

describe('agent.spawned / agent.returned events', () => {
  function supervisorConfig(): AgentConfig {
    return {
      agentId:      'supervisor',
      version:      '0.0.0',
      systemPrompt: 'system',
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' },
      subAgents: { worker: '1.0.0' },
    }
  }
  function workerConfig(): AgentConfig {
    return {
      agentId:      'worker',
      version:      '0.0.0',
      systemPrompt: 'system',
      fsm: { states: [{ name: 'react', type: 'llm' }] },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' },
    }
  }

  it('emits agent.spawned then agent.returned (completed) on the parent run', async () => {
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
    const events = await eventStore.readByRunId(result.agentRunId)

    const spawnedIdx  = events.findIndex(e => e.type === 'agent.spawned')
    const returnedIdx = events.findIndex(e => e.type === 'agent.returned')
    expect(spawnedIdx).toBeGreaterThan(-1)
    expect(returnedIdx).toBeGreaterThan(spawnedIdx)

    const spawned  = events[spawnedIdx].payload as AgentSpawnedPayload
    const returned = events[returnedIdx].payload as AgentReturnedPayload
    expect(spawned.parentRunId).toBe(result.agentRunId)
    expect(spawned.agentId).toBe('worker')
    expect(spawned.goal).toBe('subgoal')
    expect(returned.status).toBe('completed')
    expect(returned.childRunId).toBe(spawned.childRunId)
  })
})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `npx jest src/__tests__/Trace.test.ts -t "agent.spawned then agent.returned" --runInBand`
Expected: FAIL（找不到 `agent.spawned` 事件，`spawnedIdx` 为 -1）。

- [ ] **Step 3: 加两个 emit helper**

`src/runtime/AgentRuntime.ts`，在 `emitSkillLifecycle` 方法（约第 456-468 行）之后插入：

```ts
  private emitAgentSpawned(childRunId: string, agentId: string, goal: string): void {
    if (!this.eventStore) return
    // Bypass IOPort (Date.now/uuidv4 direct): informational event, not consumed
    // by replay's nondet cache. The recorder already carries the agent.spawn
    // span for the trajectory view, so this writes to the event log only.
    this.enqueueTraceWrite(async () => {
      await this.eventStore!.append({
        id:        uuidv4(),
        runId:     this.agentRunId,
        type:      'agent.spawned',
        actor:     this.config.agentId,
        timestamp: Date.now(),
        payload:   { parentRunId: this.agentRunId, childRunId, agentId, goal },
      })
    })
  }

  private emitAgentReturned(childRunId: string, status: 'completed' | 'interrupted' | 'error'): void {
    if (!this.eventStore) return
    this.enqueueTraceWrite(async () => {
      await this.eventStore!.append({
        id:        uuidv4(),
        runId:     this.agentRunId,
        type:      'agent.returned',
        actor:     this.config.agentId,
        timestamp: Date.now(),
        payload:   { childRunId, status },
      })
    })
  }
```

- [ ] **Step 4: 在 spawn 成功路径接 emit**

`src/runtime/AgentRuntime.ts` · `makeSubAgentTool`。

(a) 在首个 `await this.recordChild({ ... status: 'running' })`（约 228-233 行）之后、`try {` 之前，加：

```ts
        this.emitAgentSpawned(childContextId, agentId, goal)
```

(b) 在成功分支的 `this.recorder.recordEvent(spawnSpan, 'agent.spawn.complete', { resultStatus: result.status })`（约第 257 行）之后，加：

```ts
          this.emitAgentReturned(childContextId, result.status)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `npx jest src/__tests__/Trace.test.ts -t "agent.spawned then agent.returned" --runInBand`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add src/trace/types.ts src/runtime/AgentRuntime.ts src/__tests__/Trace.test.ts
git commit -m "feat(trace): emit agent.spawned/returned on sub-agent spawn (#24)"
```

---

### Task 3: error 态（子 run 返回 error → 走成功分支透传）+ catch 分支兜底

**Files:**
- Test: `src/__tests__/Trace.test.ts`（同一 describe 内新增）
- Modify: `src/runtime/AgentRuntime.ts`（catch 分支加 emit）

说明：子 `run()` 内部捕获错误并返回 `status:'error'`（不抛），所以 error 态走的是
**成功分支**的 `result.status` 透传——这正是本测试要锁的路径。catch 分支只在
`agentFactory.spawn` 自身抛出时触发（防御性），一并补 emit。

- [ ] **Step 1: 写失败测试（error 态）**

在 Task 2 的 describe 内追加：

```ts
  it('emits agent.returned with status error when the child run errors', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      // 只给 supervisor 的工具调用留响应；worker 的 LLM 调用拿不到 → 子 run 报错
      gateway: new StubGateway([
        toolCallResponse('s1', 'worker', { goal: 'g', input: 'i' }),
      ]),
      eventStore,
    })
    milkie.registerAgent(supervisorConfig())
    milkie.registerAgent(workerConfig())

    const result = await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })
    const events = await eventStore.readByRunId(result.agentRunId)

    const returned = events.find(e => e.type === 'agent.returned')!.payload as AgentReturnedPayload
    expect(returned.status).toBe('error')
  })
```

- [ ] **Step 2: 运行测试确认通过（成功分支已透传 result.status）**

Run: `npx jest src/__tests__/Trace.test.ts -t "status error when the child run errors" --runInBand`
Expected: PASS（Task 2 的成功分支 emit 已用 `result.status`，子 error 即透传为 'error'）。

> 若此步意外 FAIL，说明子错误走了别的路径——回到 Task 2 Step 4(b) 确认 emit 用的是 `result.status` 而非重映射值。

- [ ] **Step 3: catch 分支补防御性 emit**

`src/runtime/AgentRuntime.ts` · `makeSubAgentTool` 的 `catch (err) {` 块（约 266-275 行），在 `await this.recordChild({ ... status: 'error' })`（约 272 行）之后、`throw err` 之前，加：

```ts
          this.emitAgentReturned(childContextId, 'error')
```

- [ ] **Step 4: 运行整个 describe 确认全绿**

Run: `npx jest src/__tests__/Trace.test.ts -t "agent.spawned / agent.returned" --runInBand`
Expected: PASS（completed + error 两测试都过）。

- [ ] **Step 5: Commit**

```bash
git add src/runtime/AgentRuntime.ts src/__tests__/Trace.test.ts
git commit -m "feat(trace): emit agent.returned error on child failure + catch path (#24)"
```

---

### Task 4: 把 emitSkillLifecycle 迁到 enqueueTraceWrite（review #6 收口）

**Files:**
- Modify: `src/runtime/AgentRuntime.ts:456-468`（emitSkillLifecycle）

纯重构，行为保持：让 skill 生命周期事件也走有序 + best-effort + flush 前落盘的同一条链。由现有 skill 生命周期测试做守卫。

- [ ] **Step 1: 改 void append 为 enqueueTraceWrite**

`src/runtime/AgentRuntime.ts` · `emitSkillLifecycle`，把 `if (this.eventStore) { ... }` 块里的：

```ts
    if (this.eventStore) {
      void this.eventStore.append({
        id:        uuidv4(),
        runId:     this.agentRunId,
        type,
        actor:     this.config.agentId,
        timestamp: Date.now(),
        payload,
      })
    }
```

改为：

```ts
    if (this.eventStore) {
      this.enqueueTraceWrite(async () => {
        await this.eventStore!.append({
          id:        uuidv4(),
          runId:     this.agentRunId,
          type,
          actor:     this.config.agentId,
          timestamp: Date.now(),
          payload,
        })
      })
    }
```

（保留前面的 `this.recorder.recordEvent(this.rootSpan, type, { ...payload })` 不动。）

- [ ] **Step 2: 跑现有 skill 生命周期测试做守卫**

Run: `npx jest src/__tests__/Trace.test.ts -t "skill.loaded and skill.unloaded" --runInBand`
Expected: PASS（行为不变，事件仍按序出现在 LLM 请求前）。

- [ ] **Step 3: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "refactor(trace): route skill lifecycle through enqueueTraceWrite (#24)"
```

---

### Task 5: interrupted 态（复用 interrupt harness + eventStore）

**Files:**
- Test: `src/__tests__/AgentRuntime.test.ts`（在现有 interrupt 测试附近新增）

复用现有 supervisor interrupt harness（`SupervisorGateway` + `waitFor`），加 `eventStore` 后断言子被中断时父 run 出现 `agent.returned` status `'interrupted'`。

- [ ] **Step 1: 写失败测试**

`src/__tests__/AgentRuntime.test.ts`，在 `'propagates parent interrupt to running sub-agents and records child checkpoints'` 测试（约 374-419 行）之后插入。先确认文件顶部已 import `MemoryEventStore`（Task/历史已加）与类型 `AgentReturnedPayload`——若无，在 `from '../trace/types'` 处补 `AgentReturnedPayload`，并确保 `import { MemoryEventStore } from '../trace/MemoryEventStore'` 存在。

```ts
    it('emits agent.returned interrupted when sub-agents are interrupted', async () => {
      const stateStore = new MemoryStore()
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore,
        eventStore,
        gateway: new SupervisorGateway(),
      })

      milkie.registerAgent(makeConfig({
        agentId: 'worker-a',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'supervisor',
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
        subAgents: { 'worker-a': '1.0.0' },
      }))

      const runPromise = milkie.invoke({
        agentId:   'supervisor',
        goal:      'coordinate',
        input:     'start',
        contextId: 'ctx-int',
      })

      await waitFor(async () => {
        const children = await stateStore.get('context:ctx-int:children') as Array<{ status: string }> | undefined
        return (children ?? []).some(c => c.status === 'running')
      })

      await milkie.interrupt('ctx-int')
      const result = await runPromise

      const events = await eventStore.readByRunId(result.agentRunId)
      const returned = events
        .filter(e => e.type === 'agent.returned')
        .map(e => e.payload as AgentReturnedPayload)
      expect(returned.some(p => p.status === 'interrupted')).toBe(true)
    })
```

> 注：`SupervisorGateway` 来自本文件既有定义；`waitFor` 同。`makeConfig` 的 `subAgents` 字段已被既有 interrupt 测试使用，类型成立。

- [ ] **Step 2: 运行确认失败再通过**

Run: `npx jest src/__tests__/AgentRuntime.test.ts -t "agent.returned interrupted" --runInBand`
Expected: 若 Task 2 的成功分支 emit 已落地，则子被中断时 `result.status === 'interrupted'` 透传 → 本测试 **PASS**。

> 这是回归确认：interrupted 态与 completed/error 共用 `emitAgentReturned(childContextId, result.status)` 同一行，本测试锁住三态最后一态。

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/AgentRuntime.test.ts
git commit -m "test(trace): cover agent.returned interrupted via interrupt harness (#24)"
```

---

### Task 6: 全量回归 + 构建

**Files:** 无改动，仅校验。

- [ ] **Step 1: 构建**

Run: `npm run build`
Expected: tsc 无报错。

- [ ] **Step 2: 全量测试**

Run: `npx jest --runInBand`
Expected: 全绿（含新增 4 个测试 + 既有 region/skill/supervisor 测试不回归）。

- [ ] **Step 3: 无额外 commit（前序任务已分别提交）**

若构建/测试暴露问题，回到对应任务修复并新建修复 commit（勿 amend）。

---

## Self-Review

**Spec coverage：**
- EventKind + payload + 别名 → Task 1 ✅
- 父 spawn emit `agent.spawned`（字段正确）→ Task 2 ✅
- `agent.returned` status 与 `AgentResult.status` 一致：completed→Task 2、error→Task 3、interrupted→Task 5 ✅（三态齐）
- 走 `enqueueTraceWrite` / trace 失败不改结果 → emit helper 用 `enqueueTraceWrite`（Task 2），best-effort 由既有 `tryFlushTraceWrites` 兜底；既有「trace write failures do not change the agent result」测试在 Task 6 全量回归覆盖 ✅
- `emitSkillLifecycle` 迁移 → Task 4 ✅
- supervisor 中断/恢复 + skill 测试不回归 → Task 6 ✅
- 显式 deferral（子复用父 runId / 不 emit 子 run 生命周期 / causedBy→#30 / (2)→#47 / 验收#1放宽）→ 计划范围内不触碰，符合 spec ✅

**Placeholder scan：** 无 TBD/TODO；每个 code step 给了完整代码与精确命令。

**Type consistency：** `emitAgentSpawned(childRunId, agentId, goal)` / `emitAgentReturned(childRunId, status)` 签名在 Task 2 定义、Task 2/3 调用一致；payload 字段 `parentRunId/childRunId/agentId/goal` 与 `childRunId/status` 跨 Task 1（定义）、Task 2/3/5（断言）一致；调用处传 `childContextId` 实参对应形参 `childRunId`（语义见 spec 命名取舍 a）。
