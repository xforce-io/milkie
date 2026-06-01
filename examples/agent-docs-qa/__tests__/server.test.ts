import { startServer, stopServer } from '../server'
import type { Server } from 'http'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../../src/types/model'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { Milkie } from '../../../src/runtime/Milkie'
import { JsonlEventStore } from '../../../src/trace/JsonlEventStore'
import { FileTraceObjectStore } from '../../../src/trace/TraceObjectStore'

class StubGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('StubGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const toolCall = (name: string, args: unknown): ModelResponse => ({
  content: [], finishReason: 'tool_use', toolCalls: [{ id: 'tc-' + name, name, input: args }],
})

async function get(url: string): Promise<{ status: number; body: string }> {
  const res = await fetch(url)
  return { status: res.status, body: await res.text() }
}

async function postJson(url: string, body: unknown): Promise<{ status: number; body: string }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  return { status: res.status, body: await res.text() }
}

describe('server — REST endpoints', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string

  beforeEach(async () => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-server-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })

    server = await startServer({
      port:        0,                     // auto-assign
      exampleDir,
      gateway:     new StubGateway([text('hello from stub')]),
      agentFile:   path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot:  path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    if (!addr || typeof addr === 'string') throw new Error('server address unavailable')
    baseUrl = `http://localhost:${addr.port}`
  })

  afterEach(async () => {
    await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('GET / returns the index.html', async () => {
    const r = await get(`${baseUrl}/`)
    expect(r.status).toBe(200)
    expect(r.body).toContain('<!doctype html>')
    expect(r.body).toContain('agent playground')
  })

  it('POST /chat with no contextId mints a new one and returns runId + contextId', async () => {
    const r = await postJson(`${baseUrl}/chat`, { input: 'hi' })
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { runId: string; contextId: string; status: string }
    expect(body.runId).toMatch(/^[0-9a-f-]{36}$/)
    expect(body.contextId).toMatch(/^[0-9a-f-]{36}$/)
    expect(body.status).toBe('completed')
  })

  it('persists region content to .milkie/objects after a run', async () => {
    await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const objectsDir = path.join(exampleDir, '.milkie', 'objects')
    expect(fs.existsSync(objectsDir)).toBe(true)
    // A run composes context regions; their canonical content is written to
    // the trace object store. Non-empty objects dir proves the wiring.
    const entries = fs.readdirSync(objectsDir)
    expect(entries.length).toBeGreaterThan(0)
  })

  it('POST /chat with same contextId twice continues the same conversation', async () => {
    await stopServer(server)
    server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway([text('first'), text('second')]),
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    baseUrl = `http://localhost:${(addr as { port: number }).port}`

    const a = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q1' })).body) as { contextId: string; runId: string }
    const b = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q2', contextId: a.contextId })).body) as { contextId: string; runId: string }
    expect(b.contextId).toBe(a.contextId)
    expect(b.runId).not.toBe(a.runId)
  })

  it('GET /conversations returns empty list initially', async () => {
    const r = await get(`${baseUrl}/conversations`)
    expect(r.status).toBe(200)
    expect(JSON.parse(r.body)).toEqual({ conversations: [] })
  })

  it('GET /conversations lists prior chats after POST /chat', async () => {
    await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const r = await get(`${baseUrl}/conversations`)
    const body = JSON.parse(r.body) as { conversations: Array<{ contextId: string }> }
    expect(body.conversations).toHaveLength(1)
    expect(body.conversations[0]!.contextId).toMatch(/^[0-9a-f-]{36}$/)
  })

  it('GET /conversation/:id/events returns all events for the context in time order', async () => {
    const chat = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'hi' })).body) as { contextId: string }
    const r = await get(`${baseUrl}/conversation/${chat.contextId}/events`)
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { events: Array<{ type: string; timestamp: number }> }
    expect(body.events.length).toBeGreaterThan(0)
    const timestamps = body.events.map(e => e.timestamp)
    expect([...timestamps].sort((a, b) => a - b)).toEqual(timestamps)
  })

  it('GET /conversation/:id/events returns 404 for unknown contextId', async () => {
    const r = await get(`${baseUrl}/conversation/nonexistent/events`)
    expect(r.status).toBe(404)
  })

  it('GET /source/:relPath returns the full file when no line range', async () => {
    const r = await get(`${baseUrl}/source/chapter-01-%E6%A1%83%E5%9B%AD%E4%B8%89%E7%BB%93%E4%B9%89.txt`)
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { path: string; startLine: number; endLine: number; content: string }
    expect(body.path).toBe('chapter-01-桃园三结义.txt')
    expect(body.startLine).toBe(1)
    expect(body.endLine).toBeGreaterThan(1)
    expect(body.content.length).toBeGreaterThan(0)
  })

  it('GET /source/:relPath?lines=N-M returns the requested slice only', async () => {
    const r = await get(`${baseUrl}/source/chapter-01-%E6%A1%83%E5%9B%AD%E4%B8%89%E7%BB%93%E4%B9%89.txt?lines=33-34`)
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { startLine: number; endLine: number; content: string }
    expect(body.startLine).toBe(33)
    expect(body.endLine).toBe(34)
    expect(body.content.split('\n').length).toBe(2)
  })

  it('GET /source/:relPath?lines=N (single line) returns one line', async () => {
    const r = await get(`${baseUrl}/source/chapter-01-%E6%A1%83%E5%9B%AD%E4%B8%89%E7%BB%93%E4%B9%89.txt?lines=33`)
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { startLine: number; endLine: number; content: string }
    expect(body.startLine).toBe(33)
    expect(body.endLine).toBe(33)
    expect(body.content).not.toContain('\n')
  })

  it('GET /source/:relPath returns 404 for missing file', async () => {
    const r = await get(`${baseUrl}/source/no-such-chapter.txt`)
    expect(r.status).toBe(404)
  })

  it('GET /source/:relPath rejects path traversal (../)', async () => {
    // ../../etc/passwd attempts to escape the corpus dir.
    const r = await get(`${baseUrl}/source/..%2F..%2Fetc%2Fpasswd`)
    expect(r.status).toBe(400)
    expect(r.body).toContain('escapes corpus root')
  })

  it('GET /source/:relPath rejects malformed lines query', async () => {
    const r = await get(`${baseUrl}/source/chapter-01-%E6%A1%83%E5%9B%AD%E4%B8%89%E7%BB%93%E4%B9%89.txt?lines=abc`)
    expect(r.status).toBe(400)
  })

  it('POST /run/:runId/replay replays a recorded run deterministically', async () => {
    // Run a chat first so there's something to replay.
    const chatResp = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'hello' })).body) as { runId: string }

    const r = await postJson(`${baseUrl}/run/${chatResp.runId}/replay`, {})
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as {
      status: string; replayedOutput: string; originalOutput: string; matchesOriginal: boolean
    }
    expect(body.status).toBe('deterministic')
    expect(body.matchesOriginal).toBe(true)
    expect(body.replayedOutput).toBe('hello from stub')
    expect(body.originalOutput).toBe('hello from stub')
  })

  it('POST /run/:runId/replay returns 404 for unknown runId', async () => {
    const r = await postJson(`${baseUrl}/run/00000000-0000-0000-0000-000000000000/replay`, {})
    expect(r.status).toBe(404)
    expect(JSON.parse(r.body).status).toBe('error')
  })

  it('persists region content to .milkie/objects after a run', async () => {
    await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const objectsDir = path.join(exampleDir, '.milkie', 'objects')
    expect(fs.existsSync(objectsDir)).toBe(true)
    // A run composes context regions; their canonical content is written to
    // the trace object store. Non-empty objects dir proves the wiring.
    const entries = fs.readdirSync(objectsDir)
    expect(entries.length).toBeGreaterThan(0)
  })

  it('GET /run/:runId/viewer returns the decision viewer HTML', async () => {
    const chat = await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const { runId } = JSON.parse(chat.body) as { runId: string }

    const r = await get(`${baseUrl}/run/${runId}/viewer`)
    expect(r.status).toBe(200)
    // renderViewer emits a self-contained document with the decision spine.
    expect(r.body).toContain('<!doctype html>')
    expect(r.body).toContain('milkie trace viewer')
    expect(r.body).toContain('data-id=')        // spine nodes
    expect(r.body).toContain('spine-output')     // the output node with ❓ entry
  })

  it('GET /run/:runId/viewer 404s on an unknown run', async () => {
    const r = await get(`${baseUrl}/run/does-not-exist/viewer`)
    expect(r.status).toBe(404)
  })

  it('GET /run/:runId/execution returns the execution projection JSON', async () => {
    const chat = await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const { runId } = JSON.parse(chat.body) as { runId: string }

    const r = await get(`${baseUrl}/run/${runId}/execution`)
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as {
      steps: Array<{ kind: string; regionGroups?: Array<{ stability: string; regions: unknown[] }>; cacheHealth?: unknown }>
    }
    expect(Array.isArray(body.steps)).toBe(true)
    expect(body.steps.length).toBeGreaterThan(0)
    // The stub run makes one LLM call with composed regions → an llm step
    // whose region groups come from the core projection (not the frontend).
    const llmStep = body.steps.find(s => s.kind === 'llm')
    expect(llmStep).toBeDefined()
    expect(Array.isArray(llmStep!.regionGroups)).toBe(true)
    expect(llmStep!.regionGroups!.length).toBeGreaterThan(0)
  })

  it('GET /run/:runId/execution 404s on an unknown run', async () => {
    const r = await get(`${baseUrl}/run/does-not-exist/execution`)
    expect(r.status).toBe(404)
  })

  it('Execution tab frontend carries no attribution logic — it only renders the projection (invariant 12/13)', () => {
    const html = fs.readFileSync(path.join(__dirname, '..', 'public', 'index.html'), 'utf8')
    // #70: cache tiering, stability grouping, and event-walking attribution all
    // moved into the core buildExecutionProjection. The frontend must NOT recompute them.
    expect(html).not.toContain('classifyCacheTier')
    expect(html).not.toContain('STABILITY_ORDER')
    expect(html).not.toContain('activeRegions')
    // It consumes the core projection endpoint instead.
    expect(html).toContain('/execution`')
  })

  it('serves the audit panel with a Why tab', async () => {
    const r = await get(`${baseUrl}/`)
    expect(r.status).toBe(200)
    expect(r.body).toContain('data-tab="why"')
    expect(r.body).toContain('>Why<')
  })
})

describe('server — SSE stream', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string

  beforeEach(async () => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-sse-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })
    server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway([text('first'), text('second')]),
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    baseUrl = `http://localhost:${(addr as { port: number }).port}`
  })
  afterEach(async () => {
    await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('SSE delivers past events on connect for a completed conversation', async () => {
    // Record a conversation first
    const first = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q1' })).body) as { contextId: string }

    // Connect to SSE for this contextId
    const res = await fetch(`${baseUrl}/conversation/${first.contextId}/stream`)
    expect(res.headers.get('content-type')).toContain('text/event-stream')

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let received = ''

    const timeout = setTimeout(() => reader.cancel(), 1500)
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      received += decoder.decode(value)
    }
    clearTimeout(timeout)

    const messages = received.split('\n\n').filter(s => s.startsWith('data:'))
    expect(messages.length).toBeGreaterThan(0)
    const events = messages.map(m => JSON.parse(m.replace(/^data: /, '')))
    expect(events.some((e: { type: string }) => e.type === 'agent.run.started')).toBe(true)
    expect(events.some((e: { type: string }) => e.type === 'agent.run.completed')).toBe(true)
  }, 5_000)

  it('SSE returns empty (closed immediately) for unknown contextId', async () => {
    const res = await fetch(`${baseUrl}/conversation/nonexistent/stream`)
    expect(res.status).toBe(200)
    const text = await res.text()
    expect(text).toBe('')
  }, 5_000)
})

describe('e2e: skill loading through full chat flow', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string

  // A stub gateway that records each ModelRequest it receives so the
  // test can later inspect what the agent actually saw in its system prompt.
  class RecordingStubGateway implements IModelGateway {
    public requests: ModelRequest[] = []
    constructor(private readonly responses: ModelResponse[]) {}
    async complete(req: ModelRequest): Promise<ModelResponse> {
      this.requests.push(req)
      const r = this.responses.shift()
      if (!r) throw new Error('RecordingStubGateway exhausted')
      return r
    }
    async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
  }

  const toolUseSkillRequest = (verifierName: string): ModelResponse => ({
    content: [{ type: 'text', text: 'loading verifier for the next epoch...' }],
    toolCalls: [{
      id:    'tc-skill-1',
      name:  'skill_request',
      input: { name: verifierName },
    }],
    finishReason: 'tool_use',
  })

  const textOnly = (s: string): ModelResponse => ({
    content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
  })

  beforeEach(async () => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-skillload-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })
  })
  afterEach(async () => {
    if (server) await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('user doubt triggers skill_request → next llm.requested system prompt contains verifier', async () => {
    const stub = new RecordingStubGateway([
      toolUseSkillRequest('verifier'),       // LLM call 1: emits skill_request tool call
      textOnly('verifier loaded; final answer.'),  // LLM call 2: post-load
    ])

    server = await startServer({
      port: 0, exampleDir, gateway: stub,
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address() as { port: number }
    baseUrl = `http://localhost:${addr.port}`

    const chatResp = await postJson(`${baseUrl}/chat`, { input: '你确定吗？' })
    expect(chatResp.status).toBe(200)
    const body = JSON.parse(chatResp.body) as { contextId: string; status: string }
    expect(body.status).toBe('completed')

    // Two LLM calls were made
    expect(stub.requests).toHaveLength(2)

    // Inspect what the agent SAW in its second LLM call's system prompt —
    // this is the proof that skill loading took effect mid-invoke.
    // ModelRequest.system carries the assembled system text (base + any
    // loaded skill instructions).
    const secondSystem = stub.requests[1]!.system ?? ''

    // The verifier skill's instructions text (from agents/sanguo-researcher.md)
    // contains the literal phrase '你已进入 verifier 模式' which is highly
    // specific and won't be in the base prompt.
    expect(secondSystem).toContain('verifier')
    expect(secondSystem).toContain('你已进入 verifier 模式')

    // Sanity: the FIRST call's system should NOT contain verifier text
    // (skill loaded only after the first call's tool dispatch).
    const firstSystem = stub.requests[0]!.system ?? ''
    expect(firstSystem).not.toContain('你已进入 verifier 模式')

    // ALSO verify via the EVENT log (the trace) that the skill_request
    // tool call was recorded, since the UI reads from the event log:
    const eventsResp = await get(`${baseUrl}/conversation/${body.contextId}/events`)
    const events = (JSON.parse(eventsResp.body) as { events: Array<{ type: string; payload: unknown }> }).events
    const toolReqs = events.filter(e => e.type === 'tool.requested')
    const skillReqCalls = toolReqs.filter(e =>
      (e.payload as { toolName: string }).toolName === 'skill_request')
    expect(skillReqCalls).toHaveLength(1)
  }, 10_000)

  it('agent does NOT load verifier when user does not express doubt', async () => {
    const stub = new RecordingStubGateway([
      textOnly('here is your answer'),       // simple text-only, no tool calls
    ])

    server = await startServer({
      port: 0, exampleDir, gateway: stub,
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address() as { port: number }
    baseUrl = `http://localhost:${addr.port}`

    const chatResp = await postJson(`${baseUrl}/chat`, { input: '介绍一下赤壁之战' })
    expect(chatResp.status).toBe(200)
    const body = JSON.parse(chatResp.body) as { contextId: string }

    // Only one LLM call; no skill_request issued
    expect(stub.requests).toHaveLength(1)

    const eventsResp = await get(`${baseUrl}/conversation/${body.contextId}/events`)
    const events = (JSON.parse(eventsResp.body) as { events: Array<{ type: string; payload: unknown }> }).events
    const skillReqCalls = events.filter(e =>
      e.type === 'tool.requested' &&
      (e.payload as { toolName: string }).toolName === 'skill_request',
    )
    expect(skillReqCalls).toHaveLength(0)

    // System prompt of the single LLM call must NOT contain verifier text
    const systemText = stub.requests[0]!.system ?? ''
    expect(systemText).not.toContain('你已进入 verifier 模式')
  }, 10_000)
})

describe('diagnoser agent (stub pipeline + output contract)', () => {
  let exampleDir: string

  beforeEach(() => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-diag-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })
  })

  afterEach(() => {
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('reads the target run via tools and returns a JSON verdict', async () => {
    const toolCall = (id: string, name: string, input: unknown) => ({
      content: [{ type: 'tool_use' as const, id, name, input }],
      toolCalls: [{ id, name, input }],
      finishReason: 'tool_use' as const,
    })

    // 1) Record a target run (a normal sanguo-researcher chat) that goes off-topic.
    let server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway([text('赤壁之战发生在公元208年。')]),
      agentFile: path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address() as { port: number }
    const baseUrl = `http://localhost:${addr.port}`
    const chat = JSON.parse((await postJson(`${baseUrl}/chat`, { input: '曹操爸爸是谁' })).body) as { runId: string }
    await stopServer(server)

    // 2) Run the diagnoser against that runId.
    //    Share ONE explicit JsonlEventStore over the same runsDir so both the
    //    trace tools and the diagnoser Milkie read the same recorded events.
    const runsDir = path.join(exampleDir, '.milkie', 'runs')
    const objsDir = path.join(exampleDir, '.milkie', 'objects')
    const es = new JsonlEventStore(runsDir)
    const traceObjStore = new FileTraceObjectStore(objsDir)
    const verdict = { verdict: 'suspect', firstBreak: { step: '2', what: 'grep 赤壁', why: '与问题(曹操爸爸)不相关' }, explanation: '工具查询跑偏' }
    const milkie = new Milkie({
      eventStore: es,
      traceObjectStore: traceObjStore,
      gateway: new StubGateway([toolCall('d1', 'get_execution', { runId: chat.runId }), text(JSON.stringify(verdict))]),
    })
    milkie.loadStandardAgents()   // registers built-in diagnoser + read-Trace tools (replaces manual makeTraceTools + loadAgentFile)

    const result = await milkie.invoke({ agentId: 'diagnoser', goal: 'diagnose', input: chat.runId })
    expect(result.status).toBe('completed')
    const parsed = JSON.parse(result.output)
    expect(parsed).toHaveProperty('verdict')
    expect(parsed).toHaveProperty('firstBreak')
    expect(parsed).toHaveProperty('explanation')

    // Non-hollow assertion: verify get_execution actually executed successfully
    // and produced a real ExecutionProjection over the target run.
    const diagEvents = await es.readByRunId(result.agentRunId)
    const execResp = diagEvents.find(
      e => e.type === 'tool.responded' && (e.payload as { toolName?: string }).toolName === 'get_execution'
    )
    expect(execResp).toBeDefined()                                                              // get_execution actually ran
    const execPayload = execResp!.payload as { error?: unknown; output?: { steps?: unknown[] } }
    expect(execPayload.error).toBeUndefined()                                                   // ran successfully, not swallowed error
    expect(Array.isArray(execPayload.output?.steps)).toBe(true)                                 // produced a real ExecutionProjection
    expect(execPayload.output!.steps!.length).toBeGreaterThan(0)                                // over the (non-empty) target run
  }, 15_000)
})

describe('POST /run/:runId/diagnose', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string

  const startWith = async (gateway: IModelGateway) => {
    server = await startServer({
      port: 0, exampleDir, gateway,
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    if (!addr || typeof addr === 'string') throw new Error('server address unavailable')
    baseUrl = `http://localhost:${addr.port}`
  }

  beforeEach(() => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-diagnose-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })
  })

  afterEach(async () => {
    if (server) await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('A: suspect run returns structured verdict with a distinct diagnoseRunId', async () => {
    const verdict = {
      verdict: 'suspect',
      firstBreak: { step: 'evt-tool-1', what: "grep('赤壁')", why: '工具 query 与「曹操爸爸」无关' },
      explanation: '检索跑偏到赤壁，未回答父亲是谁。',
    }
    await startWith(new StubGateway([
      text('占位答案'),                                  // the target run's answer
      toolCall('get_run_io', { runId: 'x' }),            // diagnoser step 1
      toolCall('get_execution', { runId: 'x' }),         // diagnoser step 2
      text(JSON.stringify(verdict)),                     // diagnoser final JSON
    ]))

    const { runId } = JSON.parse((await postJson(`${baseUrl}/chat`, { input: '曹操爸爸是谁' })).body) as { runId: string }
    const r = await postJson(`${baseUrl}/run/${runId}/diagnose`, {})
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as {
      verdict: string; firstBreak: { why: string }; diagnoseRunId: string
    }
    expect(body.verdict).toBe('suspect')
    expect(body.firstBreak.why).toMatch(/赤壁|无关/)
    expect(body.diagnoseRunId).toBeTruthy()
    expect(body.diagnoseRunId).not.toBe(runId)
  }, 15_000)

  it('B: unknown run returns 404', async () => {
    await startWith(new StubGateway([]))
    const r = await postJson(`${baseUrl}/run/nope/diagnose`, {})
    expect(r.status).toBe(404)
  }, 15_000)

  it('C: non-JSON diagnoser output degrades gracefully', async () => {
    await startWith(new StubGateway([
      text('占位答案'),
      toolCall('get_run_io', { runId: 'x' }),
      toolCall('get_execution', { runId: 'x' }),
      text('这不是 JSON'),
    ]))

    const { runId } = JSON.parse((await postJson(`${baseUrl}/chat`, { input: '曹操爸爸是谁' })).body) as { runId: string }
    const r = await postJson(`${baseUrl}/run/${runId}/diagnose`, {})
    expect(r.status).toBe(200)
    expect(JSON.parse(r.body).error).toBe('unparseable')
  }, 15_000)
})
