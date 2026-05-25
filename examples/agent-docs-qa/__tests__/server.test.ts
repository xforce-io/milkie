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
