import { startServer, stopServer } from '../server'
import type { Server } from 'http'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../../src/types/model'
import fs from 'fs'
import os from 'os'
import path from 'path'

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
