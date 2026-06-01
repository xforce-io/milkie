# 内置/标准 agent 层(#89 v1)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入"内置/标准 agent(模板)"层,把 #88 的 diagnoser 从 example 上移到框架自带的顶级 `agents/`,model 由应用提供。

**Architecture:** `AgentConfig.model` 改可选;`parseConfig` 容许模板缺 model;Milkie 加 `defaultModel` opt + `resolveGateway()` helper(gatewayOverride → config.model → defaultModel → 报错)统一 4 处 gateway 决策;`loadStandardAgents()` 读包根 `agents/`(dev/prod 路径一致)并随之注册有状态的读-Trace 工具(从 example 移入 `src/tools/trace.ts`)。

**Tech Stack:** TypeScript、jest(`npx jest <path>`)、gray-matter。

测试用 jest;运行单测 `npx jest <path>`。严格 TDD:先看测试失败再实现。

---

### Task 1: `AgentConfig.model` 可选 + `parseConfig` 容许缺 model

**Files:**
- Modify: `src/types/agent.ts`(AgentConfig.model)
- Modify: `src/runtime/Milkie.ts:527`(parseConfig)
- Test: `src/__tests__/standardAgentLayer.test.ts`(新建)

- [ ] **Step 1: 写失败测试**

```typescript
// src/__tests__/standardAgentLayer.test.ts
import { Milkie } from '../runtime/Milkie'
import fs from 'fs'
import os from 'os'
import path from 'path'

function writeAgent(dir: string, name: string, body: string): string {
  const p = path.join(dir, name)
  fs.writeFileSync(p, body)
  return p
}

describe('#89 parseConfig: model optional', () => {
  let tmp: string
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-std-')) })
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }) })

  it('loads an agent template with NO model block (model is undefined)', () => {
    const file = writeAgent(tmp, 'tpl.md', `---
agentId: tpl
version: 0.0.1
fsm:
  states:
    - name: s
      type: llm
---
sys prompt`)
    const milkie = new Milkie()
    const cfg = milkie.loadAgentFile(file)
    expect(cfg.agentId).toBe('tpl')
    expect(cfg.model).toBeUndefined()
  })

  it('still throws when a model block is present but incomplete', () => {
    const file = writeAgent(tmp, 'bad.md', `---
agentId: bad
fsm:
  states:
    - name: s
      type: llm
model:
  provider: x
---
p`)
    const milkie = new Milkie()
    expect(() => milkie.loadAgentFile(file)).toThrow(/model/)
  })
})
```

- [ ] **Step 2: 跑测试,确认失败** — `npx jest src/__tests__/standardAgentLayer.test.ts` → FAIL(第一例:现 parseConfig 缺 model 抛错)。

- [ ] **Step 3: 实现** — `src/types/agent.ts` 把 `model: ModelConfig` 改为 `model?: ModelConfig`。`src/runtime/Milkie.ts` parseConfig 的 model 段改为:

```typescript
    const model = data['model'] as Record<string, string> | undefined
    if (model && (!model.provider || !model.model || !model.adapter)) {
      throw new Error('Agent config model must have provider, model, adapter')
    }
```
并把 return 里的 `model` 字段改为:
```typescript
      model: model ? {
        provider: model['provider']!,
        model:    model['model']!,
        adapter:  model['adapter']!,
        baseUrl:  model['baseUrl'] as string | undefined,
      } : undefined,
```

- [ ] **Step 4: 跑测试,确认通过** — `npx jest src/__tests__/standardAgentLayer.test.ts`。

- [ ] **Step 5: 跑 tsc 确认 model 可选未破坏其它消费点** — `npx tsc --noEmit`。若有报错(其它处假设 model 必有),在该处用 `config.model?` / 经 Task 2 的 resolveGateway 处理;只修因本改动产生的类型错误。

- [ ] **Step 6: 提交**

```bash
git add src/types/agent.ts src/runtime/Milkie.ts src/__tests__/standardAgentLayer.test.ts
git commit -m "feat(#89): AgentConfig.model optional; parseConfig tolerates template without model"
```

---

### Task 2: `defaultModel` opt + `resolveGateway()` helper(统一 4 处)

**Files:**
- Modify: `src/runtime/Milkie.ts`(MilkieOptions、构造、新 helper、invoke:184 / resume:284 / replay:391 / makeChildPort:79)
- Test: `src/__tests__/standardAgentLayer.test.ts`(追加)

- [ ] **Step 1: 写失败测试**(追加;`StubGateway` 见下)

```typescript
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'

class StubGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift(); if (!r) throw new Error('stub exhausted'); return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}
const textResp = (s: string): ModelResponse => ({ content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn' })

const NO_MODEL_AGENT = `---
agentId: nomodel
version: 0.0.1
fsm:
  states:
    - name: react
      type: llm
      on: { DONE: end }
    - name: end
      type: action
      terminal: true
---
say hi`

describe('#89 resolveGateway: no-model agent', () => {
  let tmp: string
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-rg-')) })
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }) })

  it('runs a no-model agent using the gateway override', async () => {
    const milkie = new Milkie({ gateway: new StubGateway([textResp('hi')]) })
    milkie.loadAgentFile(writeAgent(tmp, 'a.md', NO_MODEL_AGENT))
    const r = await milkie.invoke({ agentId: 'nomodel', goal: 'g', input: 'i' })
    expect(r.status).toBe('completed')
  })

  it('throws a clear error when a no-model agent has neither gateway nor defaultModel', async () => {
    const milkie = new Milkie()
    milkie.loadAgentFile(writeAgent(tmp, 'a.md', NO_MODEL_AGENT))
    await expect(milkie.invoke({ agentId: 'nomodel', goal: 'g', input: 'i' }))
      .rejects.toThrow(/no model.*gateway.*defaultModel|gateway.*defaultModel/i)
  })
})
```

- [ ] **Step 2: 跑测试,确认失败** — `npx jest src/__tests__/standardAgentLayer.test.ts -t resolveGateway` → FAIL(现 `createGateway(undefined)` 报的不是这个清晰错误 / 或第一例因 createGateway(undefined) 崩)。
  - 注:`invoke` 在顶部同步调用 `resolveGateway`(在 runtime.run 的 try/catch 之前),错误应作为 rejection 抛出。若发现 `invoke` 把它吞进 `status:'error'` 结果,则把第二例断言改为 `const r = await milkie.invoke(...); expect(r.status).toBe('error'); expect(...).toMatch(/gateway.*defaultModel/i)`,并在报告里说明。

- [ ] **Step 3: 实现** — `src/runtime/Milkie.ts`:
  - `MilkieOptions` 加 `defaultModel?: ModelConfig`(从 `../types/agent` import `ModelConfig`,若尚未 import)。
  - 字段 + 构造:`private readonly defaultModel: ModelConfig | null` 与 `this.defaultModel = opts.defaultModel ?? null`。
  - 新 helper:

```typescript
  private resolveGateway(config: AgentConfig): IModelGateway {
    if (this.gatewayOverride) return this.gatewayOverride
    const model = config.model ?? this.defaultModel ?? undefined
    if (!model) {
      throw new Error(
        `Agent "${config.agentId}" has no model and Milkie has no gateway or defaultModel; ` +
        `built-in agents need a gateway or defaultModel at construction.`)
    }
    return createGateway(model)
  }
```
  - 替换 4 处 `this.gatewayOverride ?? createGateway(config.model)` / `... createGateway(childConfig.model)` 为 `this.resolveGateway(config)` / `this.resolveGateway(childConfig)`(invoke ~184、resume ~284、replay ~391、makeChildPort ~79;makeChildPort 的闭包内确保用 `this.resolveGateway`,`this` 为 Milkie 实例)。

- [ ] **Step 4: 跑测试,确认通过** — `npx jest src/__tests__/standardAgentLayer.test.ts`(本文件全部)。

- [ ] **Step 5: 回归 + tsc** — `npx tsc --noEmit` 与 `npx jest src/__tests__/Replay.test.ts`(replay 用了 resolveGateway 路径,确认未破坏)。

- [ ] **Step 6: 提交**

```bash
git add src/runtime/Milkie.ts src/__tests__/standardAgentLayer.test.ts
git commit -m "feat(#89): defaultModel opt + resolveGateway helper (gateway/config/defaultModel fallback)"
```

---

### Task 3: 读-Trace 工具移入 `src/tools/trace.ts`(objectStore 可选)

**Files:**
- Create: `src/tools/trace.ts`
- Create: `src/__tests__/traceTools.test.ts`(从 example 测试搬入并改 import)
- (Task 6 再删 example 的旧文件)

- [ ] **Step 1: 写失败测试**

```typescript
// src/__tests__/traceTools.test.ts
import { makeTraceTools } from '../tools/trace'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import type { ToolContext } from '../types/tool'

const CTX = {} as ToolContext

async function seed(store: MemoryEventStore, runId: string) {
  await store.append({ id: 's', runId, type: 'agent.run.started', actor: 'a', timestamp: 1,
    payload: { agentId: 'x', goal: 'g', input: 'Q', contextId: runId } })
  await store.append({ id: 'lq', runId, type: 'llm.requested', actor: 'a', timestamp: 2,
    payload: { request: { model: 'm', messages: [] }, requestHash: 'h1' } })
  await store.append({ id: 'lr', runId, type: 'llm.responded', actor: 'a', timestamp: 3, causedBy: 'lq',
    payload: { response: { content: [], toolCalls: [], finishReason: 'end_turn' }, requestHash: 'h1' } })
  await store.append({ id: 'c', runId, type: 'agent.run.completed', actor: 'a', timestamp: 9,
    payload: { status: 'completed', lastTextOutput: 'A' } })
}

describe('makeTraceTools (src)', () => {
  it('get_run_io returns question + finalAnswer', async () => {
    const store = new MemoryEventStore(); await seed(store, 'r1')
    const t = makeTraceTools(store, new MemoryTraceObjectStore()).find(t => t.name === 'get_run_io')!
    expect(await t.handler({ runId: 'r1' }, CTX)).toEqual({ question: 'Q', finalAnswer: 'A' })
  })

  it('get_execution tolerates an undefined objectStore and still returns steps', async () => {
    const store = new MemoryEventStore(); await seed(store, 'r1')
    const t = makeTraceTools(store, undefined).find(t => t.name === 'get_execution')!
    const proj = await t.handler({ runId: 'r1' }, CTX) as { steps: unknown[] }
    expect(Array.isArray(proj.steps)).toBe(true)
    expect(proj.steps.length).toBeGreaterThan(0)
  })
})
```

- [ ] **Step 2: 跑测试,确认失败** — `npx jest src/__tests__/traceTools.test.ts` → FAIL(`../tools/trace` 不存在)。

- [ ] **Step 3: 实现** — 新建 `src/tools/trace.ts`,内容为 example 版 `makeTraceTools` 搬入并:(a) import 改为 src 内相对路径(去掉 `../../../src/`,改 `../trace/...`、`../types/tool`、`../trace/diagnostics/buildExecutionProjection`、`../trace/RegionContextView`);(b) 第二参 `objectStore?: ITraceObjectStore`(可选);(c) `get_execution` 的 hydrate 循环加 objectStore 守卫:

```typescript
import type { IEventStore } from '../trace/EventStore.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import type { ToolDefinition } from '../types/tool.js'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../trace/types.js'
import { regionReuseCounts } from '../trace/RegionContextView.js'
import { buildExecutionProjection } from '../trace/diagnostics/buildExecutionProjection.js'

export function makeTraceTools(
  eventStore: IEventStore,
  objectStore?: ITraceObjectStore,
): ToolDefinition[] {
  const get_run_io: ToolDefinition = {
    name: 'get_run_io',
    description: '取被诊断 run 的用户问题与最终答案。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input) => {
      const { runId } = input as { runId: string }
      const events = await eventStore.readByRunId(runId)
      let question = '', finalAnswer = ''
      for (const e of events) {
        if (e.type === 'agent.run.started') question = String((e.payload as AgentRunStartedPayload).input ?? '')
        if (e.type === 'agent.run.completed') finalAnswer = String((e.payload as AgentRunCompletedPayload).lastTextOutput ?? '')
      }
      return { question, finalAnswer }
    },
  }
  const get_execution: ToolDefinition = {
    name: 'get_execution',
    description: '取被诊断 run 的执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input) => {
      const { runId } = input as { runId: string }
      const events = await eventStore.readByRunId(runId)
      const regionContent = new Map<string, string>()
      if (objectStore) {
        for (const h of regionReuseCounts(events).keys()) {
          const c = await objectStore.getCanonical(h)
          if (c !== undefined) regionContent.set(h, c)
        }
      }
      return buildExecutionProjection(events, { regionContent })
    },
  }
  return [get_run_io, get_execution]
}
```

- [ ] **Step 4: 跑测试,确认通过** — `npx jest src/__tests__/traceTools.test.ts`。

- [ ] **Step 5: 提交**

```bash
git add src/tools/trace.ts src/__tests__/traceTools.test.ts
git commit -m "feat(#89): move read-Trace tools into src/tools/trace.ts (objectStore optional)"
```

---

### Task 4: 顶级 `agents/diagnoser.md`(无 model) + `package.json` files

**Files:**
- Create: `agents/diagnoser.md`(仓库顶级)
- Modify: `package.json`(加 `files`)

- [ ] **Step 1: 创建顶级 agent 模板** — `agents/diagnoser.md`,内容为 `examples/agent-docs-qa/agents/diagnoser.md` 的副本**但删掉整个 `model:` 块**:

```markdown
---
agentId: diagnoser
version: 0.0.1
fsm:
  states:
    - name: diagnose
      type: llm
      instructions: |
        你是一个诊断 agent。你的输入是一个被诊断 run 的 runId（一个字符串 id）。
        你的任务:判断那次 run 的最终答案是否答到了用户的问题上;若没有,沿
        「用户问题 → 工具 query → 命中证据 → 最终答案」定位第一个与问题失配的步骤。

        步骤:
        1. 调 get_run_io({ runId }) 拿到用户问题(question)和最终答案(finalAnswer)。
        2. 调 get_execution({ runId }) 拿到执行链(steps:每步 LLM/工具调用、工具的
           query、命中证据、region 组成)。
        3. 逐跳评估相关性:每个工具 query 是否针对 question?命中证据是否相关?最终
           答案是否回答了 question?
        4. 找出第一个与问题失配的跳(firstBreak)。

        最后**只输出一段 JSON**(不要任何额外文字、不要 markdown 代码围栏):
        {
          "verdict": "ok" | "suspect",
          "firstBreak": { "step": "<eventId 或步序>", "what": "<这一步做了什么>", "why": "<为什么与问题失配>" } | null,
          "explanation": "<简短中文解释>"
        }
        verdict=ok 时 firstBreak 为 null。严格只输出 JSON。
      tools: [get_run_io, get_execution]
---
诊断 agent:读被诊断 run 的 Trace 投影,定位答案与问题之间的相关性断点。
```

- [ ] **Step 2: 改 `package.json` 的 `files`** — 当前无 `files` 字段(发布会带全部);显式声明把 `dist` 与 `agents` 纳入发布产物。在 `package.json` 顶层加:

```json
  "files": ["dist", "agents"],
```
(放在 `"version"` 之后即可;确保 JSON 合法、逗号正确。)

- [ ] **Step 3: 验证 frontmatter 可被解析为"无 model 模板"** — 临时跑:`npx tsx -e "import('./src/runtime/Milkie.js').then(m=>{const k=new m.Milkie();const c=k.loadAgentFile('agents/diagnoser.md');console.log(c.agentId, c.model)})"`。Expected:输出 `diagnoser undefined`(依赖 Task 1 的 model 可选)。若报 ESM/路径问题,改用一个临时 jest 断言验证亦可。

- [ ] **Step 4: 提交**

```bash
git add agents/diagnoser.md package.json
git commit -m "feat(#89): top-level agents/diagnoser.md (model-less template) + package.json files"
```

---

### Task 5: `loadStandardAgents()`(读包根 `agents/` + 注册读-Trace 工具)

**Files:**
- Modify: `src/runtime/Milkie.ts`(新方法)
- Test: `src/__tests__/standardAgentLayer.test.ts`(追加)

- [ ] **Step 1: 写失败测试**(追加)

```typescript
import { MemoryEventStore } from '../trace/MemoryEventStore'

describe('#89 loadStandardAgents', () => {
  it('loads the built-in diagnoser and registers the read-Trace tools', () => {
    const milkie = new Milkie({ eventStore: new MemoryEventStore() })
    const ids = milkie.loadStandardAgents()
    expect(ids).toContain('diagnoser')
    expect(milkie.getAgent('diagnoser')).toBeDefined()
    expect(milkie.getAgent('diagnoser')!.model).toBeUndefined()
    // read-Trace tools registered so the diagnoser's declared tools resolve
    const toolNames = (milkie as unknown as { extraTools: Array<{ name: string }> }).extraTools.map(t => t.name)
    expect(toolNames).toEqual(expect.arrayContaining(['get_run_io', 'get_execution']))
  })
})
```

- [ ] **Step 2: 跑测试,确认失败** — `npx jest src/__tests__/standardAgentLayer.test.ts -t loadStandardAgents` → FAIL(`loadStandardAgents` 不存在)。

- [ ] **Step 3: 实现** — `src/runtime/Milkie.ts` 加方法(import `makeTraceTools` from `../tools/trace.js`):

```typescript
  /**
   * Opt-in load of milkie's built-in/standard agents (package-root `agents/`).
   * Also registers the read-Trace tools those agents depend on (when an
   * eventStore is present). Same-id agents loaded afterwards override these.
   */
  loadStandardAgents(): string[] {
    if (this.eventStore) {
      for (const t of makeTraceTools(this.eventStore, this.traceObjectStore ?? undefined)) {
        this.registerTool(t)
      }
    }
    const dir = path.join(__dirname, '..', '..', 'agents')   // src/runtime & dist/runtime both → package-root/agents
    if (!fs.existsSync(dir)) return []
    const loaded: string[] = []
    for (const f of fs.readdirSync(dir)) {
      if (f.endsWith('.md')) loaded.push(this.loadAgentFile(path.join(dir, f)).agentId)
    }
    return loaded
  }
```

- [ ] **Step 4: 跑测试,确认通过** — `npx jest src/__tests__/standardAgentLayer.test.ts`。

- [ ] **Step 5: 提交**

```bash
git add src/runtime/Milkie.ts src/__tests__/standardAgentLayer.test.ts
git commit -m "feat(#89): loadStandardAgents() loads package-root agents/ + registers read-Trace tools"
```

---

### Task 6: example 迁移 — 用内置层,删本地副本

**Files:**
- Modify: `examples/agent-docs-qa/server.ts`
- Modify: `examples/agent-docs-qa/__tests__/server.test.ts`(diagnoser 测试)
- Delete: `examples/agent-docs-qa/agents/diagnoser.md`、`examples/agent-docs-qa/tools/trace-tools.ts`
- Delete: `examples/agent-docs-qa/__tests__/trace-tools.test.ts`(已搬到 src;Task 3)

- [ ] **Step 1: 改 server.ts** — 删掉 `import { makeTraceTools } from './tools/trace-tools.js'`;把 #90 加的"注册 trace 工具循环 + 按 path 加载 diagnoser"那段替换为一行 `milkie.loadStandardAgents()`(放在 `milkie.loadAgentFile(config.agentFile)` 之后)。即:

```typescript
  for (const tool of makeCorpusToolDefinitions(config.corpusRoot)) {
    milkie.registerTool(tool)
  }
  milkie.loadAgentFile(config.agentFile)
  milkie.loadStandardAgents()   // 内置 diagnoser + 读-Trace 工具
```
并在构造 Milkie 处补 `defaultModel`,使生产(真实 LLM、无 gateway override)下无 model 的 diagnoser 可跑:

```typescript
  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    gateway:    config.gateway,
    eventStore,
    traceObjectStore,
    defaultModel: { provider: 'volcengine', model: 'doubao-seed-2-0-pro-260215', adapter: 'openai-compatible' },
  })
```

- [ ] **Step 2: 删除上移后的本地副本**

```bash
git rm examples/agent-docs-qa/agents/diagnoser.md examples/agent-docs-qa/tools/trace-tools.ts examples/agent-docs-qa/__tests__/trace-tools.test.ts
```

- [ ] **Step 3: 改 server.test.ts 的 diagnoser 测试** — 该测试原先手动 `makeTraceTools` + `loadAgentFile('../agents/diagnoser.md')`。改为走内置层:删掉对 `../tools/trace-tools` 和 `../agents/diagnoser.md` 的引用;构造诊断用的 `Milkie` 时改成 `loadStandardAgents()`,并因 stub gateway 即 override,无需 defaultModel。把该 `describe` 内的诊断 Milkie 构造与加载改为:

```typescript
    const milkie = new Milkie({
      eventStore: es,
      traceObjectStore: traceObjStore,
      gateway: new StubGateway([toolCall('d1', 'get_execution', { runId: chat.runId }), text(JSON.stringify(verdict))]),
    })
    milkie.loadStandardAgents()   // 注册内置 diagnoser + 读-Trace 工具(替代手动 makeTraceTools + loadAgentFile)
```
保留原有的非-hollow 断言(读 diagnoser run 的 `tool.responded(get_execution)` 带真实投影 + 输出 JSON 契约)不变。`import { makeTraceTools } ...` 与 diagnoser.md 路径引用一并删除。

- [ ] **Step 4: 跑 example 全套,确认通过** — `npx jest examples/agent-docs-qa/__tests__/server.test.ts`。Expected:全绿(diagnoser 经 `loadStandardAgents()` 跑通;无本地副本)。

- [ ] **Step 5: 全量回归 + tsc** — `npx tsc --noEmit`;`npx jest src/__tests__ examples/agent-docs-qa/__tests__`。Expected:全绿。

- [ ] **Step 6: 提交**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "refactor(#89): example consumes built-in diagnoser via loadStandardAgents; drop local copies"
```

---

## 完成标准

- `AgentConfig.model` 可选;`parseConfig` 模板无 model 不抛错、model 块不全仍抛错。
- `resolveGateway` 统一 4 处:override → config.model → defaultModel → 清晰错误;无 model agent 在 stub override 下可跑,在三者皆无时报清晰错。
- 读-Trace 工具在 `src/tools/trace.ts`、容忍 objectStore undefined。
- `agents/diagnoser.md`(顶级、无 model);`package.json` files 含 `dist`+`agents`。
- `loadStandardAgents()` 加载内置 diagnoser + 注册读-Trace 工具。
- example 改用 `loadStandardAgents()` + `defaultModel`,删除本地 diagnoser.md / trace-tools.ts / trace-tools.test.ts;诊断 stub 管道测仍非-hollow 且通过。
- `tsc` 干净;`src/__tests__` + example 测试全绿,无回归(尤其 Replay)。
