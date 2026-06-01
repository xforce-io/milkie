# diagnoser agent 实现计划(#88)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `examples/agent-docs-qa` 内做一个专职 `diagnoser` agent,读被诊断 run 的 Trace 投影,沿「问题→工具query→证据→答案」定位第一个相关性断点。

**Architecture:** 确定性的读-Trace 工具(包装核心 `buildExecutionProjection`,可纯单测)+ 一个 LLM diagnoser agent(方法论写在 systemPrompt,输出约定 JSON 契约)。PoC 借住 example,归宿见 #89。

**Tech Stack:** TypeScript、jest、milkie SDK(`Milkie` / `ToolDefinition` / `MemoryEventStore` / `MemoryTraceObjectStore` / `buildExecutionProjection`)。

**测试策略(关键)**:读-Trace 工具是确定性的 → 严格 TDD。diagnoser 的**判断质量**依赖真实 LLM,stub 下只能验"管道通 + 输出契约成立",判断质量归 live e2e(手动,不进 CI)。

---

### Task 1: 读-Trace 工具骨架 + `get_run_io`

**Files:**
- Create: `examples/agent-docs-qa/tools/trace-tools.ts`
- Test: `examples/agent-docs-qa/__tests__/trace-tools.test.ts`

- [ ] **Step 1: 写失败测试**

```typescript
// examples/agent-docs-qa/__tests__/trace-tools.test.ts
import { makeTraceTools } from '../tools/trace-tools'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../../../src/trace/TraceObjectStore'
import type { ToolContext } from '../../../src/types/tool'

const CTX = {} as ToolContext  // 这些工具不使用 ctx

async function seedRun(store: MemoryEventStore, runId: string) {
  await store.append({ id: 's', runId, type: 'agent.run.started', actor: 'a', timestamp: 1,
    payload: { agentId: 'x', goal: 'g', input: '曹操爸爸是谁', contextId: runId } })
  await store.append({ id: 'c', runId, type: 'agent.run.completed', actor: 'a', timestamp: 9,
    payload: { status: 'completed', lastTextOutput: '赤壁之战发生在公元208年。' } })
}

describe('makeTraceTools: get_run_io', () => {
  it('returns the user question and final answer of a run', async () => {
    const store = new MemoryEventStore()
    await seedRun(store, 'target-1')
    const tools = makeTraceTools(store, new MemoryTraceObjectStore())
    const getRunIo = tools.find(t => t.name === 'get_run_io')!
    const out = await getRunIo.handler({ runId: 'target-1' }, CTX)
    expect(out).toEqual({ question: '曹操爸爸是谁', finalAnswer: '赤壁之战发生在公元208年。' })
  })
})
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `npx jest examples/agent-docs-qa/__tests__/trace-tools.test.ts`
Expected: FAIL — `Cannot find module '../tools/trace-tools'`.

- [ ] **Step 3: 写最小实现**

```typescript
// examples/agent-docs-qa/tools/trace-tools.ts
import type { IEventStore } from '../../../src/trace/EventStore'
import type { ITraceObjectStore } from '../../../src/trace/TraceObjectStore'
import type { ToolDefinition } from '../../../src/types/tool'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../../../src/trace/types'

/**
 * Read-Trace tools for the diagnoser agent: deterministic projections over a
 * recorded run's event log. Wrap core projections; never parse raw events in
 * the agent. `runId` is the run being DIAGNOSED (distinct from the diagnoser's
 * own run).
 */
export function makeTraceTools(
  eventStore: IEventStore,
  _objectStore: ITraceObjectStore,
): ToolDefinition[] {
  const get_run_io: ToolDefinition = {
    name: 'get_run_io',
    description: '取被诊断 run 的用户问题与最终答案。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input) => {
      const { runId } = input as { runId: string }
      const events = await eventStore.readByRunId(runId)
      let question = ''
      let finalAnswer = ''
      for (const e of events) {
        if (e.type === 'agent.run.started') question = String((e.payload as AgentRunStartedPayload).input ?? '')
        if (e.type === 'agent.run.completed') finalAnswer = String((e.payload as AgentRunCompletedPayload).lastTextOutput ?? '')
      }
      return { question, finalAnswer }
    },
  }

  return [get_run_io]
}
```

- [ ] **Step 4: 跑测试,确认通过**

Run: `npx jest examples/agent-docs-qa/__tests__/trace-tools.test.ts`
Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add examples/agent-docs-qa/tools/trace-tools.ts examples/agent-docs-qa/__tests__/trace-tools.test.ts
git commit -m "feat(#88): get_run_io read-Trace tool"
```

---

### Task 2: `get_execution` 工具(包装 buildExecutionProjection)

**Files:**
- Modify: `examples/agent-docs-qa/tools/trace-tools.ts`
- Test: `examples/agent-docs-qa/__tests__/trace-tools.test.ts`

- [ ] **Step 1: 写失败测试**(追加到 trace-tools.test.ts)

```typescript
describe('makeTraceTools: get_execution', () => {
  it('returns the execution projection (steps with tool query) of a run', async () => {
    const store = new MemoryEventStore()
    const runId = 'target-2'
    await store.append({ id: 's', runId, type: 'agent.run.started', actor: 'a', timestamp: 1,
      payload: { agentId: 'x', goal: 'g', input: '曹操爸爸是谁', contextId: runId } })
    await store.append({ id: 'lq', runId, type: 'llm.requested', actor: 'a', timestamp: 2,
      payload: { request: { model: 'm', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }, requestHash: 'h1' } })
    await store.append({ id: 'lr', runId, type: 'llm.responded', actor: 'a', timestamp: 3, causedBy: 'lq',
      payload: { response: { content: [], toolCalls: [], finishReason: 'tool_use' }, requestHash: 'h1' } })
    await store.append({ id: 'tq', runId, type: 'tool.requested', actor: 'a', timestamp: 4,
      payload: { toolName: 'grep', input: { pattern: '赤壁' }, requestHash: 'h2' } })
    await store.append({ id: 'tr', runId, type: 'tool.responded', actor: 'a', timestamp: 5, causedBy: 'tq',
      payload: { toolName: 'grep', output: { matches: [] }, requestHash: 'h2' } })

    const tools = makeTraceTools(store, new MemoryTraceObjectStore())
    const getExec = tools.find(t => t.name === 'get_execution')!
    const proj = await getExec.handler({ runId }, CTX) as { steps: Array<{ kind: string; tool?: { name: string; input: unknown } }> }
    const toolStep = proj.steps.find(s => s.kind === 'tool')
    expect(toolStep?.tool).toMatchObject({ name: 'grep', input: { pattern: '赤壁' } })
  })
})
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `npx jest examples/agent-docs-qa/__tests__/trace-tools.test.ts -t get_execution`
Expected: FAIL — `getExec` is undefined (`get_execution` not in the tool list).

- [ ] **Step 3: 写最小实现**(在 trace-tools.ts 顶部加 import,新增工具并加入返回数组)

```typescript
// 顶部 imports 追加:
import { regionReuseCounts } from '../../../src/trace/RegionContextView'
import { buildExecutionProjection } from '../../../src/trace/diagnostics/buildExecutionProjection'
```

```typescript
// 在 get_run_io 之后、return 之前新增:
const get_execution: ToolDefinition = {
  name: 'get_execution',
  description: '取被诊断 run 的执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。入参 { runId }。',
  inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
  handler: async (input) => {
    const { runId } = input as { runId: string }
    const events = await eventStore.readByRunId(runId)
    // Hydrate region content the same way `trace report` / #70 endpoint does.
    const regionContent = new Map<string, string>()
    for (const h of regionReuseCounts(events).keys()) {
      const c = await _objectStore.getCanonical(h)
      if (c !== undefined) regionContent.set(h, c)
    }
    return buildExecutionProjection(events, { regionContent })
  },
}
```

```typescript
// 把 return 改成:
return [get_run_io, get_execution]
```

(注:`_objectStore` 现在被使用,可把参数名从 `_objectStore` 改为 `objectStore`。)

- [ ] **Step 4: 跑测试,确认全部通过**

Run: `npx jest examples/agent-docs-qa/__tests__/trace-tools.test.ts`
Expected: PASS（get_run_io + get_execution 两组）。

- [ ] **Step 5: 提交**

```bash
git add examples/agent-docs-qa/tools/trace-tools.ts examples/agent-docs-qa/__tests__/trace-tools.test.ts
git commit -m "feat(#88): get_execution read-Trace tool wrapping buildExecutionProjection"
```

---

### Task 3: diagnoser agent 定义

**Files:**
- Create: `examples/agent-docs-qa/agents/diagnoser.md`

- [ ] **Step 1: 写 agent 定义**(prompt 工程,无单测;由 Task 4 的管道测覆盖)

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
model:
  provider: volcengine
  model: doubao-seed-2-0-pro-260215
  adapter: openai-compatible
---
诊断 agent:读被诊断 run 的 Trace 投影,定位答案与问题之间的相关性断点。
```

- [ ] **Step 2: 提交**

```bash
git add examples/agent-docs-qa/agents/diagnoser.md
git commit -m "feat(#88): diagnoser agent definition (methodology + JSON output contract)"
```

---

### Task 4: server 接入 + stub 管道/契约测

**Files:**
- Modify: `examples/agent-docs-qa/server.ts:281-284`
- Test: `examples/agent-docs-qa/__tests__/server.test.ts`

- [ ] **Step 1: 写失败测试**(追加到 server.test.ts 的 "server — REST endpoints" describe 之外,新 describe)

```typescript
describe('diagnoser agent (stub pipeline + output contract)', () => {
  it('diagnoser reads the target run via tools and returns a JSON verdict', async () => {
    const fs = require('fs'), os = require('os'), path = require('path')
    const exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-diag-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })

    // Stub: 1) diagnoser calls get_execution, 2) emits the JSON verdict.
    const toolCall = (id: string, name: string, input: unknown) => ({
      content: [{ type: 'tool_use', id, name, input }], toolCalls: [{ id, name, input }], finishReason: 'tool_use',
    })
    const text = (s: string) => ({ content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn' })

    // First record a target run with sanguo-researcher (a normal chat).
    let server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway([text('赤壁之战发生在公元208年。')]),
      agentFile: path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    let addr = server.address() as { port: number }
    let baseUrl = `http://localhost:${addr.port}`
    const chat = JSON.parse((await postJson(`${baseUrl}/chat`, { input: '曹操爸爸是谁' })).body) as { runId: string }
    await stopServer(server)

    // Now run the diagnoser against that runId via the SDK directly.
    const { Milkie } = require('../../../src/runtime/Milkie')
    const { JsonlEventStore } = require('../../../src/trace/JsonlEventStore')
    const { FileTraceObjectStore } = require('../../../src/trace/TraceObjectStore')
    const { makeTraceTools } = require('../tools/trace-tools')
    const verdict = { verdict: 'suspect', firstBreak: { step: '2', what: 'grep 赤壁', why: '与问题(曹操爸爸)不相关' }, explanation: '工具查询跑偏' }
    const milkie = new Milkie({
      eventStore: new JsonlEventStore(path.join(exampleDir, '.milkie', 'runs')),
      traceObjectStore: new FileTraceObjectStore(path.join(exampleDir, '.milkie', 'objects')),
      gateway: new StubGateway([toolCall('d1', 'get_execution', { runId: chat.runId }), text(JSON.stringify(verdict))]),
    })
    for (const t of makeTraceTools((milkie as any).eventStore ?? new JsonlEventStore(path.join(exampleDir, '.milkie', 'runs')),
                                   new FileTraceObjectStore(path.join(exampleDir, '.milkie', 'objects')))) {
      milkie.registerTool(t)
    }
    milkie.loadAgentFile(path.join(__dirname, '..', 'agents', 'diagnoser.md'))

    const result = await milkie.invoke({ agentId: 'diagnoser', goal: 'diagnose', input: chat.runId })
    expect(result.status).toBe('completed')
    const parsed = JSON.parse(result.output)
    expect(parsed).toHaveProperty('verdict')
    expect(parsed).toHaveProperty('firstBreak')
    expect(parsed).toHaveProperty('explanation')

    fs.rmSync(exampleDir, { recursive: true, force: true })
  }, 15_000)
})
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts -t "diagnoser agent"`
Expected: FAIL — diagnoser agentId 未注册 / 工具未注册(取决于 stub 路径)。
（若失败信息提示 `makeTraceTools` 的 eventStore 取值方式不稳,改为显式 `const es = new JsonlEventStore(path.join(exampleDir,'.milkie','runs'))` 复用同一实例。）

- [ ] **Step 3: 让 server 默认也注册 trace 工具 + 加载 diagnoser**(生产路径,使 panel 集成 issue 后续可直接用)

修改 `examples/agent-docs-qa/server.ts`,在 `for (const tool of makeCorpusToolDefinitions(...))` 之后追加:

```typescript
import { makeTraceTools } from './tools/trace-tools.js'   // 顶部 import 区追加

// ...在 milkie.loadAgentFile(config.agentFile) 之前:
for (const tool of makeTraceTools(eventStore, traceObjectStore)) {
  milkie.registerTool(tool)
}
// 同目录的 diagnoser(若存在则加载,使其与领域 agent 并存)
const diagnoserPath = path.join(path.dirname(config.agentFile), 'diagnoser.md')
if (existsSync(diagnoserPath)) milkie.loadAgentFile(diagnoserPath)
```

- [ ] **Step 4: 跑测试,确认通过**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts -t "diagnoser agent"`
Expected: PASS — diagnoser 跑通、output 可 `JSON.parse` 成契约(verdict/firstBreak/explanation)。

- [ ] **Step 5: 跑整套 server 测,确认无回归**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts`
Expected: 全绿(现有 + 新增 diagnoser 用例)。

- [ ] **Step 6: 提交**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(#88): wire trace tools + diagnoser into server; stub pipeline/contract test"
```

---

### Task 5: README 标注 + live e2e 说明

**Files:**
- Modify: `examples/agent-docs-qa/README.md`

- [ ] **Step 1: 在 README 加一节**

```markdown
## diagnoser agent(#88,借住)

`agents/diagnoser.md` 是一个**横切诊断 agent**(答案错因诊断):读被诊断 run 的 Trace
投影(`tools/trace-tools.ts`),沿「问题→工具query→证据→答案」定位第一个相关性断点。

- 它**借住**于本 example;归宿是 milkie 的内置/标准 agent 层(见 issue #89),不属于"三国问答"领域。
- 编程入口:`milkie.invoke({ agentId: 'diagnoser', input: <被诊断的 runId> })`,输出为
  JSON `{ verdict, firstBreak, explanation }`。
- 确定性测试覆盖读-Trace 工具 + 管道/契约;**诊断判断质量**需真实 LLM,见下方 live 验证。

### live 验证(手动,需 VOLCENGINE_TOKEN/API_BASE)

启动 server,问一个会跑偏的问题(stub 或真实),拿到 runId 后:
`milkie.invoke({ agentId: 'diagnoser', input: '<runId>' })` —— 人工核对 `firstBreak`
是否指向真正跑偏的那一步(例:问"曹操爸爸"却 grep"赤壁")。
```

- [ ] **Step 2: 提交**

```bash
git add examples/agent-docs-qa/README.md
git commit -m "docs(#88): README — diagnoser agent (borrowed; home #89) + live verification"
```

---

## 完成标准

- 读-Trace 工具(`get_run_io` / `get_execution`)有确定性单测,全绿。
- diagnoser agent 经 stub 管道测:被 invoke、调工具拿到被诊断 run 投影、输出能 `JSON.parse` 成契约。
- 整套 example 测试无回归。
- README 标注借住 + 归宿 #89 + live 验证步骤。
- (手动,非 CI)真实 doubao 对"曹操爸爸→grep赤壁"诊断,`firstBreak` 指向工具 query 跑偏。
