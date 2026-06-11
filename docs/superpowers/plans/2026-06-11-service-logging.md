# Service Logging (#79) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 milkie 增加结构化服务日志层（pino，NDJSON 到 stderr），覆盖 HTTP 请求、invoke 汇总、LLM 调用、生命周期与既有零散 console.* 的收编。

**Architecture:** 新建 `src/logging/`（pino 工厂 + 类型别名 + LoggingGateway 装饰器），通过 pino child 注入 `mod`/`runId`/`contextId` 维度；埋点只落在服务边界（serve 分发层、`Milkie.invoke`、`GatewayFactory.createGateway` 包装点）。设计定稿见 `docs/design/79-service-logging.md`。

**Tech Stack:** TypeScript（CJS 编译）、pino ^10、pino-pretty ^13（devDep）、jest + ts-jest。

**约定:** 所有命令在 `/Users/xupeng/dev/github/milkie` 下执行，分支 `feat/79-service-logging`。每个任务 TDD：先写测试看它失败，再最小实现，再全绿提交。

---

### Task 1: logger 工厂（`src/logging/logger.ts`）

**Files:**
- Create: `src/logging/logger.ts`
- Create: `src/logging/index.ts`
- Test: `src/__tests__/logging.test.ts`
- Modify: `package.json`（依赖）

- [ ] **Step 1: 安装依赖**

```bash
npm install pino@^10.0.0 && npm install -D pino-pretty@^13.0.0
```

- [ ] **Step 2: 写失败测试**

```ts
// src/__tests__/logging.test.ts
import { createServiceLogger, resolveFormat, getLogger, setLogger } from '../logging/logger'

/** 内存 sink：每条 NDJSON 一行，parse 后断言字段。 */
function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

describe('createServiceLogger', () => {
  it('emits NDJSON with ts(ISO8601)/level-label/msg', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    log.info({ port: 8080 }, 'serve listening')
    const [line] = sink.lines()
    expect(line.msg).toBe('serve listening')
    expect(line.level).toBe('info')                                   // 字符串标签而非数字
    expect(line.port).toBe(8080)
    expect(typeof line.ts).toBe('string')
    expect(() => new Date(line.ts as string).toISOString()).not.toThrow()
    expect(line.pid).toBeUndefined()                                  // base 字段去掉
    expect(line.hostname).toBeUndefined()
  })

  it('filters below configured level', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'warn', format: 'json', destination: sink.stream })
    log.info('hidden')
    log.warn('shown')
    expect(sink.lines().map(l => l.msg)).toEqual(['shown'])
  })

  it('level=silent emits nothing', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'silent', format: 'json', destination: sink.stream })
    log.error('nope')
    expect(sink.lines()).toEqual([])
  })

  it('child loggers merge mod and correlation ids', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    log.child({ mod: 'server' }).child({ runId: 'r1', contextId: 'c1' }).info('http request')
    const [line] = sink.lines()
    expect(line.mod).toBe('server')
    expect(line.runId).toBe('r1')
    expect(line.contextId).toBe('c1')
  })

  it('serializes err with message and stack', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    log.error({ err: new Error('boom') }, 'llm call failed')
    const [line] = sink.lines()
    const err = line.err as { message: string; stack: string }
    expect(err.message).toBe('boom')
    expect(err.stack).toContain('Error: boom')
  })

  it('defaults level from LOG_LEVEL env', () => {
    const prev = process.env['LOG_LEVEL']
    process.env['LOG_LEVEL'] = 'error'
    try {
      const sink = memorySink()
      const log = createServiceLogger({ format: 'json', destination: sink.stream })
      log.warn('hidden')
      log.error('shown')
      expect(sink.lines().map(l => l.msg)).toEqual(['shown'])
    } finally {
      if (prev === undefined) delete process.env['LOG_LEVEL']
      else process.env['LOG_LEVEL'] = prev
    }
  })
})

describe('resolveFormat', () => {
  it('LOG_FORMAT=json overrides TTY', () => expect(resolveFormat({ LOG_FORMAT: 'json' }, true)).toBe('json'))
  it('LOG_FORMAT=pretty overrides non-TTY', () => expect(resolveFormat({ LOG_FORMAT: 'pretty' }, false)).toBe('pretty'))
  it('defaults to pretty on TTY', () => expect(resolveFormat({}, true)).toBe('pretty'))
  it('defaults to json off TTY', () => expect(resolveFormat({}, false)).toBe('json'))
  it('ignores unknown LOG_FORMAT values', () => expect(resolveFormat({ LOG_FORMAT: 'xml' }, false)).toBe('json'))
})

describe('getLogger / setLogger', () => {
  afterEach(() => setLogger(undefined))
  it('getLogger returns a stable lazy singleton', () => {
    expect(getLogger()).toBe(getLogger())
  })
  it('setLogger swaps the singleton (test injection); setLogger(undefined) resets', () => {
    const sink = memorySink()
    const injected = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    setLogger(injected)
    expect(getLogger()).toBe(injected)
    setLogger(undefined)
    expect(getLogger()).not.toBe(injected)
  })
})
```

- [ ] **Step 3: 跑测试确认失败**

Run: `npx jest src/__tests__/logging.test.ts --runInBand`
Expected: FAIL —— `Cannot find module '../logging/logger'`

- [ ] **Step 4: 最小实现**

```ts
// src/logging/logger.ts
import pino from 'pino'

/**
 * #79 服务日志（service log）：运维遥测层，与 event log（业务真相）、trajectory
 * （span 诊断）三层并存，用 runId/contextId 关联。边界纪律、字段 schema 见
 * docs/design/79-service-logging.md。
 *
 * 类型上只 re-export pino.Logger —— 不在 pino 外造抽象层，调用方不直接 import pino。
 */
export type ServiceLogger = pino.Logger

export interface ServiceLoggerOptions {
  level?:       string                    // 默认 LOG_LEVEL ?? 'info'
  format?:      'json' | 'pretty'         // 默认 LOG_FORMAT ?? (stderr TTY ? pretty : json)
  destination?: pino.DestinationStream    // 默认 stderr；测试注入内存 sink
}

/** format 决策独立成纯函数，TTY/env 分支可单测。 */
export function resolveFormat(env: { LOG_FORMAT?: string }, isTTY: boolean): 'json' | 'pretty' {
  if (env.LOG_FORMAT === 'pretty') return 'pretty'
  if (env.LOG_FORMAT === 'json') return 'json'
  return isTTY ? 'pretty' : 'json'
}

export function createServiceLogger(opts: ServiceLoggerOptions = {}): ServiceLogger {
  const level  = opts.level ?? process.env['LOG_LEVEL'] ?? 'info'
  const format = opts.format ?? resolveFormat(
    { LOG_FORMAT: process.env['LOG_FORMAT'] },
    process.stderr.isTTY === true,
  )
  // 日志走 stderr：stdout 已被程序输出占用（CLI 结果、MILKIE_SERVE_READY 端口标记）。
  let destination: pino.DestinationStream | undefined = opts.destination
  if (!destination && format === 'pretty') {
    try {
      // pino-pretty 是 devDependency；生产未安装时静默回退 json（即 undefined → 下方 stderr）
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const pretty = require('pino-pretty') as (o: object) => pino.DestinationStream
      destination = pretty({ destination: 2 })
    } catch { /* fall back to json on stderr */ }
  }
  destination ??= pino.destination(2)
  return pino({
    level,
    base: undefined,                                            // 去掉 pid/hostname
    timestamp: () => `,"ts":"${new Date().toISOString()}"`,     // 字段名用 ts（设计 §4）
    formatters: { level: label => ({ level: label }) },         // 字符串标签而非数字
    serializers: { err: pino.stdSerializers.err },
  }, destination)
}

let defaultLogger: ServiceLogger | undefined

/** 惰性默认单例（读环境变量）。模块级埋点（如 tools/system.ts）从这里取。 */
export function getLogger(): ServiceLogger {
  defaultLogger ??= createServiceLogger()
  return defaultLogger
}

/** 测试注入用：换掉单例；传 undefined 恢复惰性默认。 */
export function setLogger(logger: ServiceLogger | undefined): void {
  defaultLogger = logger
}
```

```ts
// src/logging/index.ts
export { createServiceLogger, getLogger, setLogger, resolveFormat } from './logger.js'
export type { ServiceLogger, ServiceLoggerOptions } from './logger.js'
export { LoggingGateway } from './LoggingGateway.js'   // Task 2 落地前先注释本行
```

注意：pino 自定义 `timestamp` 输出 `ts` 字段后，pino 默认的 `time` 字段不再出现；
若 `line.ts` 断言失败而出现 `time`，检查 timestamp 函数返回值必须以 `,` 开头。

- [ ] **Step 5: 跑测试确认通过**

Run: `npx jest src/__tests__/logging.test.ts --runInBand`
Expected: PASS（全部用例）

- [ ] **Step 6: 提交**

```bash
git add package.json package-lock.json src/logging src/__tests__/logging.test.ts
git commit -m "feat(#79): 服务日志 logger 工厂 —— pino/NDJSON-stderr/ts-ISO 字段/级别与 format 决策"
```

---

### Task 2: LoggingGateway 装饰器 + GatewayFactory 接线

**Files:**
- Create: `src/logging/LoggingGateway.ts`
- Modify: `src/gateway/GatewayFactory.ts`
- Test: `src/__tests__/LoggingGateway.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
// src/__tests__/LoggingGateway.test.ts
import { LoggingGateway } from '../logging/LoggingGateway'
import { createServiceLogger } from '../logging/logger'
import { createGateway } from '../gateway/GatewayFactory'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

const REQ: ModelRequest = { model: 'test-model', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }

function okGateway(): IModelGateway {
  return {
    async complete(): Promise<ModelResponse> {
      return {
        content: [{ type: 'text', text: 'ok' }], toolCalls: [],
        usage: { inputTokens: 11, outputTokens: 7 }, finishReason: 'end_turn',
      }
    },
    async *stream(): AsyncIterable<ModelEvent> {
      yield { type: 'message_delta', data: { text: 'ok' } }
      yield { type: 'usage', data: { inputTokens: 3, outputTokens: 5 } }
    },
  }
}

function failGateway(): IModelGateway {
  return {
    async complete(): Promise<ModelResponse> { throw new Error('rate limited') },
    // eslint-disable-next-line require-yield
    async *stream(): AsyncIterable<ModelEvent> { throw new Error('socket reset') },
  }
}

describe('LoggingGateway.complete', () => {
  it('logs one info wide event with model/durationMs/tokens on success', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(okGateway(), log)
    await gw.complete(REQ)
    const [line] = sink.lines()
    expect(line.msg).toBe('llm call')
    expect(line.level).toBe('info')
    expect(line.model).toBe('test-model')
    expect(typeof line.durationMs).toBe('number')
    expect(line.inputTokens).toBe(11)
    expect(line.outputTokens).toBe(7)
  })

  it('logs error with err.stack and rethrows on failure', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(failGateway(), log)
    await expect(gw.complete(REQ)).rejects.toThrow('rate limited')
    const [line] = sink.lines()
    expect(line.level).toBe('error')
    expect(line.msg).toBe('llm call failed')
    expect((line.err as { message: string }).message).toBe('rate limited')
  })

  it('does not log prompt or completion content (脱敏)', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    await new LoggingGateway(okGateway(), log).complete(REQ)
    const serialized = JSON.stringify(sink.lines())
    expect(serialized).not.toContain('"hi"')   // prompt 正文
    expect(serialized).not.toContain('"ok"')   // completion 正文
  })
})

describe('LoggingGateway.stream', () => {
  it('passes events through and logs one summary with accumulated usage', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(okGateway(), log)
    const events: ModelEvent[] = []
    for await (const e of gw.stream(REQ)) events.push(e)
    expect(events).toHaveLength(2)                       // 事件原样透传
    const [line] = sink.lines()
    expect(line.msg).toBe('llm stream')
    expect(line.model).toBe('test-model')
    expect(line.inputTokens).toBe(3)
    expect(line.outputTokens).toBe(5)
  })

  it('logs error and rethrows when the stream throws', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(failGateway(), log)
    await expect(async () => { for await (const _ of gw.stream(REQ)) { /* drain */ } }).rejects.toThrow('socket reset')
    const [line] = sink.lines()
    expect(line.level).toBe('error')
    expect(line.msg).toBe('llm stream failed')
  })
})

describe('createGateway wiring', () => {
  it('returns a LoggingGateway-wrapped adapter', () => {
    const gw = createGateway({ provider: 'anthropic', model: 'claude-x', adapter: 'anthropic' })
    expect(gw).toBeInstanceOf(LoggingGateway)
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx jest src/__tests__/LoggingGateway.test.ts --runInBand`
Expected: FAIL —— `Cannot find module '../logging/LoggingGateway'`

- [ ] **Step 3: 实现 LoggingGateway 并接线 GatewayFactory**

```ts
// src/logging/LoggingGateway.ts
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model.js'
import type { ServiceLogger } from './logger.js'

/**
 * #79：在 gateway 统一包装点打 LLM wide event（每调用一条：model/durationMs/token），
 * 两个 adapter 一处覆盖。只记元数据，不携带 prompt/completion 正文（脱敏，设计 §6）。
 */
export class LoggingGateway implements IModelGateway {
  constructor(
    private readonly inner: IModelGateway,
    private readonly log:   ServiceLogger,
  ) {}

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const startedAt = Date.now()
    try {
      const res = await this.inner.complete(request)
      this.log.info({
        model: request.model, durationMs: Date.now() - startedAt,
        inputTokens: res.usage?.inputTokens, outputTokens: res.usage?.outputTokens,
      }, 'llm call')
      return res
    } catch (err) {
      this.log.error({ model: request.model, durationMs: Date.now() - startedAt, err }, 'llm call failed')
      throw err
    }
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelEvent> {
    const startedAt = Date.now()
    let inputTokens = 0
    let outputTokens = 0
    try {
      for await (const e of this.inner.stream(request)) {
        if (e.type === 'usage') {
          inputTokens  += e.data.inputTokens
          outputTokens += e.data.outputTokens
        }
        yield e
      }
      this.log.info({ model: request.model, durationMs: Date.now() - startedAt, inputTokens, outputTokens }, 'llm stream')
    } catch (err) {
      this.log.error({ model: request.model, durationMs: Date.now() - startedAt, err }, 'llm stream failed')
      throw err
    }
  }
}
```

注意：`ModelEvent` 的 usage 分支形如 `{ type: 'usage', data: { inputTokens, outputTokens } }`
（见 `src/gateway/AnthropicAdapter.ts:198-202`）；若类型收窄报错，按 `src/types/model.ts`
的实际判别联合写 type guard。

```ts
// src/gateway/GatewayFactory.ts —— 全文替换为：
import type { ModelConfig } from '../types/agent.js'
import type { IModelGateway } from '../types/model.js'
import { AnthropicAdapter } from './AnthropicAdapter.js'
import { OpenAICompatibleAdapter } from './OpenAICompatibleAdapter.js'
import { LoggingGateway } from '../logging/LoggingGateway.js'
import { getLogger, type ServiceLogger } from '../logging/logger.js'

export function createGateway(model: ModelConfig, logger?: ServiceLogger): IModelGateway {
  const adapter = model.adapter.toLowerCase()
  // #79：统一在工厂出口包 LoggingGateway，两个 adapter 一处覆盖。
  // 注入的 gateway（MilkieOptions.gateway，测试用）不经过这里，因此不被包装。
  const log = logger ?? getLogger().child({ mod: 'gateway' })

  if (adapter === 'anthropic') {
    return new LoggingGateway(new AnthropicAdapter({ baseUrl: model.baseUrl }), log)
  }

  if (
    adapter === 'openai-compatible' ||
    adapter === 'openai' ||
    adapter === 'volcengine'
  ) {
    return new LoggingGateway(new OpenAICompatibleAdapter({
      baseUrl: model.baseUrl ?? process.env['VOLCENGINE_API_BASE'],
      apiKey:
        process.env['VOLCENGINE_TOKEN'] ??
        process.env['OPENAI_API_KEY'],
    }), log)
  }

  throw new Error(`Unknown model adapter: "${model.adapter}"`)
}
```

同时把 `src/logging/index.ts` 中 Task 1 注释掉的 `LoggingGateway` export 行放开。

- [ ] **Step 4: 跑测试确认通过**

Run: `npx jest src/__tests__/LoggingGateway.test.ts src/__tests__/logging.test.ts --runInBand`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/logging src/gateway/GatewayFactory.ts src/__tests__/LoggingGateway.test.ts
git commit -m "feat(#79): LoggingGateway —— gateway 工厂出口统一打 LLM wide event(model/耗时/token/err)"
```

---

### Task 3: `Milkie.invoke` 汇总日志 + tier 回退收编

**Files:**
- Modify: `src/runtime/Milkie.ts`
- Test: `src/__tests__/Milkie.logging.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
// src/__tests__/Milkie.logging.test.ts
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { createServiceLogger } from '../logging/logger'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

const AGENT: AgentConfig = {
  agentId: 'echo', version: '1.0.0', systemPrompt: 'echo',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
}

function okGateway(): IModelGateway {
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      yield { type: 'message_delta', data: { text: 'done' } }
    },
  }
}

function failGateway(): IModelGateway {
  return {
    async complete(): Promise<ModelResponse> { throw new Error('provider down') },
    // eslint-disable-next-line require-yield
    async *stream(): AsyncIterable<ModelEvent> { throw new Error('provider down') },
  }
}

describe('Milkie.invoke service log', () => {
  it('logs one info summary per invoke: mod/agentId/runId/contextId/durationMs/status', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: okGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    const result = await milkie.invoke({ agentId: 'echo', goal: 'g', input: 'hi', contextId: 'ctx-1' })
    const summaries = sink.lines().filter(l => l.msg === 'invoke completed')
    expect(summaries).toHaveLength(1)
    const [line] = summaries
    expect(line.mod).toBe('runtime')
    expect(line.agentId).toBe('echo')
    expect(line.runId).toBe(result.agentRunId)
    expect(line.contextId).toBe('ctx-1')
    expect(line.status).toBe('completed')
    expect(typeof line.durationMs).toBe('number')
  })

  it('concurrent invokes keep their own runId/contextId（不串线）', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: okGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    const [a, b] = await Promise.all([
      milkie.invoke({ agentId: 'echo', goal: 'g', input: 'x', contextId: 'ctx-a' }),
      milkie.invoke({ agentId: 'echo', goal: 'g', input: 'y', contextId: 'ctx-b' }),
    ])
    const byCtx = Object.fromEntries(
      sink.lines().filter(l => l.msg === 'invoke completed').map(l => [l.contextId, l.runId]),
    )
    expect(byCtx['ctx-a']).toBe(a.agentRunId)
    expect(byCtx['ctx-b']).toBe(b.agentRunId)
  })

  it('LLM 持续失败时仍出一条 invoke 汇总（AgentRuntime 把错误吞成 status:error）', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: failGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    const result = await milkie.invoke({ agentId: 'echo', goal: 'g', input: 'hi', contextId: 'ctx-err' })
    expect(result.status).toBe('error')
    const summaries = sink.lines().filter(l => typeof l.msg === 'string' && (l.msg as string).startsWith('invoke'))
    expect(summaries).toHaveLength(1)
    expect(summaries[0].status).toBe('error')
  })

  it('tier 回退从 console.debug 收编为 logger.warn', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: okGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    await milkie.complete('echo', { messages: [{ role: 'user', content: [{ type: 'text', text: 'q' }] }], tier: 'no-such-tier' })
    const warns = sink.lines().filter(l => l.level === 'warn')
    expect(warns.length).toBeGreaterThanOrEqual(1)
    expect(warns[0].tier).toBe('no-such-tier')
    expect(warns[0].agentId).toBe('echo')
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx jest src/__tests__/Milkie.logging.test.ts --runInBand`
Expected: FAIL —— `logger` 不是 MilkieOptions 的属性（TS 编译错）或断言无日志行

- [ ] **Step 3: 实现**

`src/runtime/Milkie.ts` 改动四处：

(a) import 区新增：

```ts
import { getLogger, type ServiceLogger } from '../logging/logger.js'
```

(b) `MilkieOptions` 增加可选注入（追加在 `traceObjectStore?` 之后）：

```ts
  /** #79：服务日志 logger；缺省用进程级 getLogger()。测试注入内存 sink。 */
  logger?:          ServiceLogger
```

类字段与构造器对应增加：

```ts
  private readonly log: ServiceLogger
  // constructor 内：
  this.log = opts.logger ?? getLogger()
```

(c) `resolveModel` 的 console.debug 行（约 :116）替换为：

```ts
    if (tier) this.log.warn({ mod: 'runtime', agentId: config.agentId, tier }, 'tier not found; falling back to default model')
```

(d) `invoke()` 末段（`const rec = …` 之前）建 child 并计时，try/catch 两路都打汇总。
现 `invoke()` 尾部：

```ts
    const rec = ioPort instanceof RecordingIOPort ? ioPort : null
    await rec?.attach({ … })

    try {
      const result = await runtime.run(request.input)
      await rec?.detach({ status: result.status, lastTextOutput: result.output })
      return result
    } catch (err) {
      await rec?.detach({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      throw err
    }
```

改为：

```ts
    const rec = ioPort instanceof RecordingIOPort ? ioPort : null
    await rec?.attach({
      agentId:   config.agentId,
      goal:      request.goal,
      input:     request.input,
      contextId,
      ...(previousRunId ? { previousRunId } : {}),
    })

    // #79：每 invoke 一条服务日志 wide event（边界汇总，设计 §5）。
    // turns/token 不在 AgentResult 上，token 已由 LoggingGateway 按调用记录。
    const invokeLog = this.log.child({ mod: 'runtime', runId: agentRunId, contextId })
    const invokeStartedAt = Date.now()
    try {
      const result = await runtime.run(request.input)
      await rec?.detach({ status: result.status, lastTextOutput: result.output })
      invokeLog.info({ agentId: config.agentId, durationMs: Date.now() - invokeStartedAt, status: result.status }, 'invoke completed')
      return result
    } catch (err) {
      await rec?.detach({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      invokeLog.error({ agentId: config.agentId, durationMs: Date.now() - invokeStartedAt, err }, 'invoke failed')
      throw err
    }
```

(e) `resolveGateway` 把 logger 传给工厂，注入的 logger 才能贯通 gateway 层：

```ts
    return createGateway(model, this.log.child({ mod: 'gateway' }))
```

(f) 测试静默：不注入 logger 的存量测试会落到默认 `getLogger()` 朝 stderr 打 info 噪音。
新建 `jest.setup.ts` 并在 `jest.config.ts` 注册：

```ts
// jest.setup.ts —— 测试进程默认静默服务日志；显式注入 logger 的用例不受影响。
process.env['LOG_LEVEL'] ??= 'silent'
```

```ts
// jest.config.ts 的 config 对象内追加：
  setupFiles: ['<rootDir>/jest.setup.ts'],
```

- [ ] **Step 4: 跑测试确认通过（含回归）**

Run: `npx jest src/__tests__/Milkie.logging.test.ts --runInBand && npm run test:unit`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add src/runtime/Milkie.ts src/__tests__/Milkie.logging.test.ts
git commit -m "feat(#79): Milkie.invoke 每次一条服务日志汇总；tier 回退收编为 logger.warn"
```

---

### Task 4: serve HTTP 请求日志 + 生命周期日志

**Files:**
- Modify: `src/cli/serve.ts`
- Test: `src/__tests__/serve.logging.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
// src/__tests__/serve.logging.test.ts
import http from 'http'
import type { AddressInfo } from 'net'
import { createServeServer } from '../cli/serve'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { BroadcastingEventStore } from '../trace/BroadcastingEventStore'
import { createServiceLogger } from '../logging/logger'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

function buildMilkie(): { milkie: Milkie; agentId: string; broadcaster: BroadcastingEventStore } {
  const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
  const gateway: IModelGateway = {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      yield { type: 'message_delta', data: { text: 'ok' } }
    },
  }
  const agent: AgentConfig = {
    agentId: 'echo', version: '1.0.0', systemPrompt: 'echo',
    fsm: { states: [{ name: 'react', type: 'llm' }] },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  }
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: broadcaster, gateway })
  milkie.registerAgent(agent)
  return { milkie, agentId: 'echo', broadcaster }
}

function listen(server: http.Server): Promise<number> {
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve((server.address() as AddressInfo).port)))
}

function get(port: number, path: string): Promise<number> {
  return new Promise((resolve, reject) => {
    http.get({ host: '127.0.0.1', port, path }, res => { res.resume(); res.on('end', () => resolve(res.statusCode ?? 0)) }).on('error', reject)
  })
}

function post(port: number, path: string, body: unknown): Promise<{ status: number; text: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, path, method: 'POST', headers: { 'content-type': 'application/json' } },
      res => {
        let text = ''
        res.on('data', c => { text += c })
        res.on('end', () => resolve({ status: res.statusCode ?? 0, text }))
      },
    )
    req.on('error', reject)
    req.end(JSON.stringify(body))
  })
}

/** res.on('finish') 的日志写入在响应抵达客户端后才跑，轮询等它出现。 */
async function waitFor<T>(read: () => T | undefined, ms = 2000): Promise<T> {
  const deadline = Date.now() + ms
  for (;;) {
    const v = read()
    if (v !== undefined) return v
    if (Date.now() > deadline) throw new Error('timed out waiting for log line')
    await new Promise(r => setTimeout(r, 10))
  }
}

describe('serve HTTP service log', () => {
  let server: http.Server
  afterEach(() => new Promise<void>(r => server.close(() => r())))

  it('logs one wide event per request: mod/method/path/status/ms', async () => {
    const sink = memorySink()
    const { milkie, agentId, broadcaster } = buildMilkie()
    server = createServeServer({
      milkie, agentId, broadcaster,
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    const port = await listen(server)
    expect(await get(port, '/health')).toBe(200)
    const line = await waitFor(() => sink.lines().find(l => l.msg === 'http request'))
    expect(line.mod).toBe('server')
    expect(line.method).toBe('GET')
    expect(line.path).toBe('/health')
    expect(line.status).toBe(200)
    expect(typeof line.ms).toBe('number')
  })

  it('logs 404 and 400 statuses (异常路径同样有 wide event)', async () => {
    const sink = memorySink()
    const { milkie, agentId, broadcaster } = buildMilkie()
    server = createServeServer({
      milkie, agentId, broadcaster,
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    const port = await listen(server)
    await get(port, '/no-such-route')
    await post(port, '/chat', {})                       // 缺 contextId → 400
    const seen = await waitFor(() => {
      const ls = sink.lines().filter(l => l.msg === 'http request')
      return ls.length >= 2 ? ls : undefined
    })
    expect(seen.map(l => [l.path, l.status])).toEqual(expect.arrayContaining([
      ['/no-such-route', 404],
      ['/chat', 400],
    ]))
  })

  it('SSE 请求记录完整流时长（/chat 正常流）', async () => {
    const sink = memorySink()
    const { milkie, agentId, broadcaster } = buildMilkie()
    server = createServeServer({
      milkie, agentId, broadcaster,
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    const port = await listen(server)
    const res = await post(port, '/chat', { contextId: 'c1', input: 'hi' })
    expect(res.status).toBe(200)
    expect(res.text).toContain('agent.run.completed')
    const line = await waitFor(() => sink.lines().find(l => l.msg === 'http request' && l.path === '/chat'))
    expect(line.status).toBe(200)
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx jest src/__tests__/serve.logging.test.ts --runInBand`
Expected: FAIL —— `logger` 不是 ServeOptions 属性（TS 编译错）

- [ ] **Step 3: 实现**

`src/cli/serve.ts` 改动：

(a) import 新增：

```ts
import { getLogger, type ServiceLogger } from '../logging/logger.js'
```

(b) `ServeOptions` 追加：

```ts
  /** #79：服务日志 logger；缺省用进程级 getLogger()。测试注入内存 sink。 */
  logger?:     ServiceLogger
```

(c) `createServeServer` 开头取 child，并在 `http.createServer` 回调打请求 wide event：

```ts
export function createServeServer(opts: ServeOptions): Server {
  const { milkie, agentId, broadcaster } = opts
  const log = (opts.logger ?? getLogger()).child({ mod: 'server' })
```

```ts
  return http.createServer((req, res) => {
    // #79：每请求一条 wide event。挂在 res 'finish' 上，SSE 长连接记录的是完整流时长。
    const startedAt = Date.now()
    res.on('finish', () => {
      log.info({
        method: req.method,
        path:   new URL(req.url ?? '/', 'http://localhost').pathname,
        status: res.statusCode,
        ms:     Date.now() - startedAt,
      }, 'http request')
    })
    void (async () => {
      …（原有路由分发体不动）…
    })()
  })
```

(d) `runServeServer` 加可选 logger 参数并打生命周期日志：

```ts
export function runServeServer(server: Server, opts: { port: number; host?: string; logger?: ServiceLogger }): Promise<void> {
  const log = (opts.logger ?? getLogger()).child({ mod: 'server' })
```

`listen` 回调内（readiness 标记行之后）：

```ts
      log.info({ port: addr.port, host: opts.host ?? '127.0.0.1' }, 'serve listening')
```

`shutdown` 函数内（`closing = true` 之后）：

```ts
      log.info('serve shutting down')
```

(e) `serveMain` 打启动配置摘要并传递 logger：

```ts
export async function serveMain(opts: { agent: string; port: number; host?: string; stateStore?: 'memory' | 'sqlite'; dataDir?: string }): Promise<void> {
  const log = getLogger().child({ mod: 'server' })
  const { stateStore, eventStore } = await buildServeStores(opts)
  const broadcaster = new BroadcastingEventStore(eventStore)
  const milkie = new Milkie({ stateStore, eventStore: broadcaster })
  const config = milkie.loadAgentFile(opts.agent)
  log.info({ agentId: config.agentId, stateStore: opts.stateStore ?? 'memory', dataDir: opts.dataDir }, 'serve starting')
  const server = createServeServer({ milkie, agentId: config.agentId, broadcaster })
  await runServeServer(server, { port: opts.port, host: opts.host })
}
```

- [ ] **Step 4: 跑测试确认通过（含 serve 回归）**

Run: `npx jest src/__tests__/serve.logging.test.ts src/__tests__/serve.test.ts src/__tests__/serve-persistence.test.ts src/__tests__/serveCli.test.ts --runInBand`
Expected: 全 PASS（serveCli 测试若解析 stdout，确认日志在 stderr 未污染它）

- [ ] **Step 5: 提交**

```bash
git add src/cli/serve.ts src/__tests__/serve.logging.test.ts
git commit -m "feat(#79): serve 每请求 wide event(method/path/status/ms) + 启动/关闭生命周期日志"
```

---

### Task 5: `tools/system.ts` 三处 console.warn 收编

**Files:**
- Modify: `src/tools/system.ts`
- Modify: `src/__tests__/skillListManifest.test.ts`（现有 console.warn spy 断言迁移）

- [ ] **Step 1: 改造现有测试为 logger 注入（先行，作为失败测试）**

`src/__tests__/skillListManifest.test.ts`：

把顶部的 spy 设施

```ts
let warnSpy: jest.SpyInstance
// beforeEach 内
warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {})
// afterEach 内
warnSpy.mockRestore()
```

替换为 setLogger 注入（import 区加 `import { createServiceLogger, setLogger } from '../logging/logger'`）：

```ts
let warnLines: Record<string, unknown>[]
// beforeEach 内：
warnLines = []
const raw: string[] = []
setLogger(createServiceLogger({
  level: 'warn', format: 'json',
  destination: { write: (s: string) => {
    raw.push(s)
    warnLines.length = 0
    warnLines.push(...raw.flatMap(x => x.split('\n').filter(Boolean)).map(x => JSON.parse(x) as Record<string, unknown>))
  } },
}))
// afterEach 内：
setLogger(undefined)
```

断言对应替换：
- `expect(warnSpy).toHaveBeenCalled()` → `expect(warnLines.length).toBeGreaterThan(0)`
- `expect(warnSpy).not.toHaveBeenCalled()` → `expect(warnLines).toHaveLength(0)`

并新增一条字段断言（任选一个已有的"manifest 读取失败"用例内加）：

```ts
    expect(warnLines[0].mod).toBe('tools')
    expect(warnLines[0].level).toBe('warn')
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx jest src/__tests__/skillListManifest.test.ts --runInBand`
Expected: FAIL —— system.ts 仍走 console.warn，warnLines 为空

- [ ] **Step 3: 实现**

`src/tools/system.ts`：import 区加 `import { getLogger } from '../logging/logger.js'`，
`loadSkillManifest()` 开头取 `const log = getLogger().child({ mod: 'tools' })`，三处替换：

```ts
log.warn({ manifestPath, err: e as Error }, `skill_list: failed to read ${SKILL_MANIFEST_ENV}`)
```

```ts
log.warn({ manifestPath }, `skill_list: ${SKILL_MANIFEST_ENV} parsed but has no valid 'skills' array; treating as unconfigured`)
```

```ts
log.warn({ entry: JSON.stringify(s) }, 'skill_list: skipping malformed skill entry (missing name/description)')
```

（原 `[milkie]` 前缀去掉——`mod: 'tools'` 字段已承担来源标识。）

- [ ] **Step 4: 跑测试确认通过**

Run: `npx jest src/__tests__/skillListManifest.test.ts src/__tests__/toolOverrideContract.test.ts --runInBand`
Expected: PASS（toolOverrideContract 如有 console spy 同步处理）

- [ ] **Step 5: 提交**

```bash
git add src/tools/system.ts src/__tests__/skillListManifest.test.ts
git commit -m "refactor(#79): tools/system.ts 零散 console.warn 收编进服务日志(mod=tools)"
```

---

### Task 6: 全量回归 + lint + 文档/issue 收口 + PR

**Files:**
- Verify only（无新代码）；`docs/design/79-service-logging.md` 已随分支提交

- [ ] **Step 1: 全量验证**

```bash
npm run lint && npm run build && npm test
```

Expected: lint 0 error；tsc 通过；单测 + 确定性 e2e 全 PASS。
若 e2e 中有解析 stdout 的用例失败，检查是否有日志误写 stdout（必须全在 stderr）。

- [ ] **Step 2: 提交设计文档与计划**

```bash
git add docs/design/79-service-logging.md docs/superpowers/plans/2026-06-11-service-logging.md
git commit -m "docs(#79): 服务日志设计文档与实现计划"
```

- [ ] **Step 3: 推送并建 PR**

```bash
git push -u origin feat/79-service-logging
gh pr create --repo xforce-io/milkie --title "feat(#79): 结构化服务日志（pino/NDJSON-stderr，三层可观测之 logging 层）" --body "$(cat <<'EOF'
Closes #79。设计文档：docs/design/79-service-logging.md（讨论记录见 issue 评论）。

- src/logging/：pino 工厂（ts/level/mod/err schema，LOG_LEVEL/LOG_FORMAT/TTY）+ LoggingGateway
- 埋点：serve 每请求 wide event、Milkie.invoke 汇总、LLM 调用（工厂出口统一包装）、启动/关闭
- 收编：tools/system.ts 三处 console.warn、Milkie tier 回退 console.debug
- 脱敏：不携带 prompt/completion 正文；与 event log 以 runId 关联
- 偏离 issue 原文两点（已在设计文档 §2/§5 说明）：日志走 stderr（stdout 被 CLI 结果与
  MILKIE_SERVE_READY 占用）；invoke 汇总暂无 turns 字段（AgentResult 未暴露）

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: issue 正文收敛为摘要 + 设计文档链接（repo doc 即真相源）**
