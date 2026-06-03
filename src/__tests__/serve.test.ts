import http from 'http'
import { createServeServer } from '../cli/serve'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { BroadcastingEventStore } from '../trace/BroadcastingEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import type { ToolDefinition } from '../types/tool'

// ─────────────────────────── test harnesses ───────────────────────────

/**
 * A Milkie whose single agent streams `deltas` as token-level message_delta
 * events in one LLM turn, then finishes. Drives the streaming path when invoke
 * is given onModelEvent.
 */
function buildTextMilkie(deltas: string[]): { milkie: Milkie; agentId: string; broadcaster: BroadcastingEventStore } {
  const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
  const full = deltas.join('')
  const gateway: IModelGateway = {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text: full }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      for (const d of deltas) yield { type: 'message_delta', data: { text: d } }
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

/**
 * A Milkie that loops `totalSteps` LLM turns, each issuing one `work_step`
 * tool call that sleeps `stepMs` — long enough for an interrupt to land between
 * turns. Mirrors examples/interrupt-resume-sidecar's deterministic stepper.
 */
function buildSteppingMilkie(opts: { totalSteps?: number; stepMs?: number } = {}): { milkie: Milkie; agentId: string; broadcaster: BroadcastingEventStore } {
  const totalSteps = opts.totalSteps ?? 6
  const stepMs = opts.stepMs ?? 60
  const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
  let counter = 0

  const workStep: ToolDefinition = {
    name: 'work_step', description: 'one unit of work',
    inputSchema: { type: 'object', properties: { stepId: { type: 'number' } }, required: ['stepId'] },
    handler: async (input: unknown) => {
      const { stepId } = input as { stepId: number }
      await new Promise<void>(r => setTimeout(r, stepMs))
      return { stepId, done: true }
    },
  }
  // serve always drives the streaming path (invoke is given onModelEvent), so the
  // stepper must emit its work_step calls as a tool_call stream — an empty stream
  // would aggregate to "no tool calls" and finish the run immediately.
  const nextRound = (): { done: boolean; stepId: number } => {
    if (counter >= totalSteps) return { done: true, stepId: 0 }
    counter++
    return { done: false, stepId: counter }
  }
  const gateway: IModelGateway = {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      const { done, stepId } = nextRound()
      if (done) return { content: [{ type: 'text', text: 'all done' }], toolCalls: [], finishReason: 'end_turn' }
      return { content: [{ type: 'tool_use', id: `w${stepId}`, name: 'work_step', input: { stepId } }], toolCalls: [{ id: `w${stepId}`, name: 'work_step', input: { stepId } }], finishReason: 'tool_use' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      const { done, stepId } = nextRound()
      if (done) { yield { type: 'message_delta', data: { text: 'all done' } }; return }
      yield { type: 'tool_call_start', data: { toolCallId: `w${stepId}`, name: 'work_step' } }
      yield { type: 'tool_call_done',  data: { toolCallId: `w${stepId}`, input: { stepId } } }
    },
  }
  const agent: AgentConfig = {
    agentId: 'stepper', version: '1.0.0', systemPrompt: 'step until done',
    fsm: { states: [{ name: 'react', type: 'llm', max_iterations: totalSteps + 5 }] },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  }
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: broadcaster, gateway, tools: [workStep] })
  milkie.registerAgent(agent)
  return { milkie, agentId: 'stepper', broadcaster }
}

/** A Milkie whose gateway throws, to exercise the error path (invoke rejects). */
function buildErrorMilkie(): { milkie: Milkie; agentId: string; broadcaster: BroadcastingEventStore } {
  const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
  const gateway: IModelGateway = {
    async complete(_req: ModelRequest): Promise<ModelResponse> { throw new Error('kaboom') },
    async *stream(_req: ModelRequest): AsyncIterable<never> { throw new Error('kaboom') },
  }
  const agent: AgentConfig = {
    agentId: 'boomer', version: '1.0.0', systemPrompt: 'boom',
    fsm: { states: [{ name: 'react', type: 'llm' }] },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  }
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: broadcaster, gateway })
  milkie.registerAgent(agent)
  return { milkie, agentId: 'boomer', broadcaster }
}

// ─────────────────────────── http helpers ───────────────────────────

const servers: http.Server[] = []
function listen(server: http.Server): Promise<number> {
  servers.push(server)
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => {
    resolve((server.address() as { port: number }).port)
  }))
}
afterEach(() => { for (const s of servers.splice(0)) s.close() })

function request(port: number, method: string, path: string, body?: unknown): Promise<{ status: number; json: unknown }> {
  return new Promise((resolve, reject) => {
    const data = body !== undefined ? JSON.stringify(body) : undefined
    const req = http.request(
      { host: '127.0.0.1', port, method, path, headers: data ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) } : {} },
      res => { let buf = ''; res.on('data', c => { buf += c }); res.on('end', () => resolve({ status: res.statusCode!, json: buf ? JSON.parse(buf) : null })) },
    )
    req.on('error', reject)
    if (data) req.write(data)
    req.end()
  })
}

interface SSEEvent { event: string; data: unknown }
function parseFrame(raw: string): SSEEvent | null {
  let event = 'message'; const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return null
  return { event, data: JSON.parse(dataLines.join('\n')) }
}
/** Open an SSE request; resolve with all frames once the stream closes. Also exposes the live request (to fire interrupts mid-stream). */
function sse(port: number, method: string, path: string, body?: unknown): { done: Promise<SSEEvent[]>; req: http.ClientRequest } {
  const data = body !== undefined ? JSON.stringify(body) : undefined
  let req!: http.ClientRequest
  const done = new Promise<SSEEvent[]>((resolve, reject) => {
    req = http.request(
      { host: '127.0.0.1', port, method, path, headers: data ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) } : {} },
      res => {
        const events: SSEEvent[] = []
        let buf = ''
        res.on('data', c => {
          buf += c
          let idx
          while ((idx = buf.indexOf('\n\n')) >= 0) {
            const frame = parseFrame(buf.slice(0, idx)); buf = buf.slice(idx + 2)
            if (frame) events.push(frame)
          }
        })
        res.on('end', () => resolve(events))
        res.on('error', reject)
      },
    )
    req.on('error', reject)
    if (data) req.write(data)
    req.end()
  })
  return { done, req }
}

// ─────────────────────────────── tests ───────────────────────────────

describe('createServeServer', () => {
  it('GET /health returns { ok: true }', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['hi'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'GET', '/health')
    expect(res.status).toBe(200)
    expect(res.json).toEqual({ ok: true })
  })

  it('POST /chat streams token-level message_delta then a completed terminal event', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['Hello, ', 'world!'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const { done } = sse(port, 'POST', '/chat', { contextId: 'c1', input: 'hi' })
    const events = await done

    const deltas = events.filter(e => e.event === 'message_delta')
    expect(deltas.map(e => (e.data as { text: string }).text)).toEqual(['Hello, ', 'world!'])

    const terminal = events.find(e => e.event === 'agent.run.completed')
    expect(terminal).toBeDefined()
    expect(terminal!.data).toMatchObject({ status: 'completed', output: 'Hello, world!' })
  })

  it('POST /chat without contextId returns 400', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/chat', { input: 'hi' })
    expect(res.status).toBe(400)
    expect(res.json).toMatchObject({ error: expect.stringContaining('contextId') })
  })

  it('POST /interrupt ends the running /chat stream with an interrupted terminal (not a bare disconnect)', async () => {
    const { milkie, agentId, broadcaster } = buildSteppingMilkie({ totalSteps: 30, stepMs: 40 })
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const { done } = sse(port, 'POST', '/chat', { contextId: 'ic', input: 'go' })
    await new Promise(r => setTimeout(r, 120))   // let a few steps run
    const ir = await request(port, 'POST', '/interrupt', { contextId: 'ic' })
    expect(ir.status).toBe(200)
    expect(ir.json).toMatchObject({ signaled: true })

    const events = await done
    const terminal = events.find(e => e.event === 'agent.run.completed')
    expect(terminal).toBeDefined()
    expect((terminal!.data as { status: string }).status).toBe('interrupted')
  })

  it('POST /resume continues an interrupted run to completion on a fresh stream', async () => {
    const { milkie, agentId, broadcaster } = buildSteppingMilkie({ totalSteps: 4, stepMs: 40 })
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))

    const { done: chatDone } = sse(port, 'POST', '/chat', { contextId: 'rc', input: 'go' })
    await new Promise(r => setTimeout(r, 70))
    await request(port, 'POST', '/interrupt', { contextId: 'rc' })
    const chatEvents = await chatDone
    expect((chatEvents.find(e => e.event === 'agent.run.completed')!.data as { status: string }).status).toBe('interrupted')

    const { done: resumeDone } = sse(port, 'POST', '/resume', { contextId: 'rc' })
    const resumeEvents = await resumeDone
    const terminal = resumeEvents.find(e => e.event === 'agent.run.completed')
    expect(terminal).toBeDefined()
    expect((terminal!.data as { status: string }).status).toBe('completed')
  })

  it('POST /chat surfaces a failing run as an error frame + error terminal (not a silent disconnect)', async () => {
    const { milkie, agentId, broadcaster } = buildErrorMilkie()
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const { done } = sse(port, 'POST', '/chat', { contextId: 'ec', input: 'go' })
    const events = await done

    const terminal = events.find(e => e.event === 'agent.run.completed')
    expect(terminal).toBeDefined()
    expect((terminal!.data as { status: string }).status).toBe('error')
  })

  it('a client disconnecting mid-stream does not crash the server', async () => {
    const { milkie, agentId, broadcaster } = buildSteppingMilkie({ totalSteps: 20, stepMs: 30 })
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))

    const { done, req } = sse(port, 'POST', '/chat', { contextId: 'disc', input: 'go' })
    done.catch(() => { /* client aborts on purpose */ })
    await new Promise(r => setTimeout(r, 100))   // a few steps in
    req.destroy()                                 // client disconnects mid-stream
    await new Promise(r => setTimeout(r, 250))    // run keeps going; would-be writes fire at the dead socket

    // the server must still be alive and serving
    const res = await request(port, 'GET', '/health')
    expect(res.status).toBe(200)
    expect(res.json).toEqual({ ok: true })
  })

  it('POST /context/set then /context/get round-trips a variable', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const setRes = await request(port, 'POST', '/context/set', { contextId: 'cv', name: 'model_name', value: 'claude' })
    expect(setRes.status).toBe(200)
    expect(setRes.json).toMatchObject({ ok: true })
    const getRes = await request(port, 'POST', '/context/get', { contextId: 'cv', name: 'model_name' })
    expect(getRes.json).toMatchObject({ value: 'claude' })
  })

  it('POST /context/get returns null for a missing variable', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/context/get', { contextId: 'cv', name: 'nope' })
    expect(res.status).toBe(200)
    expect(res.json).toMatchObject({ value: null })
  })

  it('POST /context/list returns all variables for a context', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    await request(port, 'POST', '/context/set', { contextId: 'cv', name: 'a', value: '1' })
    await request(port, 'POST', '/context/set', { contextId: 'cv', name: 'b', value: '2' })
    const res = await request(port, 'POST', '/context/list', { contextId: 'cv' })
    expect(res.json).toMatchObject({ vars: { a: '1', b: '2' } })
  })

  it('POST /context/set without contextId returns 400', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/context/set', { name: 'k', value: 'v' })
    expect(res.status).toBe(400)
  })

  // ─────────────────────────── #124 /llm ───────────────────────────

  const userMsg = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]

  it('POST /llm (non-streaming) returns the aggregated output as JSON', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['Sum', 'mary'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/llm', { messages: userMsg })
    expect(res.status).toBe(200)
    expect(res.json).toMatchObject({ output: 'Summary' })
  })

  it('POST /llm with stream=true streams token-level message_delta then a done terminal', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['a', 'b', 'c'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const { done } = sse(port, 'POST', '/llm', { messages: userMsg, stream: true })
    const events = await done
    const deltas = events.filter(e => e.event === 'message_delta')
    expect(deltas.map(e => (e.data as { text: string }).text)).toEqual(['a', 'b', 'c'])
    expect(events.find(e => e.event === 'done')).toBeDefined()
  })

  it('POST /llm without messages returns 400', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/llm', { system: 'only' })
    expect(res.status).toBe(400)
    expect(res.json).toMatchObject({ error: expect.stringContaining('messages') })
  })

  it('POST /llm with stream=true surfaces a failing gateway as an error frame + done', async () => {
    const { milkie, agentId, broadcaster } = buildErrorMilkie()
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const { done } = sse(port, 'POST', '/llm', { messages: userMsg, stream: true })
    const events = await done
    expect(events.find(e => e.event === 'error')).toBeDefined()
    expect(events.find(e => e.event === 'done')).toBeDefined()
  })

  // ──────────────────────── #124 /session ────────────────────────

  it('POST /session/export returns a portable session after a chat', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['ok'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    await (await sse(port, 'POST', '/chat', { contextId: 'se', input: 'hi' }).done)
    const res = await request(port, 'POST', '/session/export', { contextId: 'se' })
    expect(res.status).toBe(200)
    expect(res.json).toMatchObject({ manifest: { schemaVersion: 1, contextId: 'se', agentId } })
    expect(Array.isArray((res.json as { events: unknown[] }).events)).toBe(true)
  })

  it('POST /session/export without contextId returns 400', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/session/export', {})
    expect(res.status).toBe(400)
  })

  it('POST /session/export for an unknown context returns 404', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const res = await request(port, 'POST', '/session/export', { contextId: 'never' })
    expect(res.status).toBe(404)
  })

  it('POST /session/import round-trips an exported session into a fresh server', async () => {
    const a = buildTextMilkie(['ok'])
    const portA = await listen(createServeServer(a))
    await (await sse(portA, 'POST', '/chat', { contextId: 'rt', input: 'hi' }).done)
    const exported = (await request(portA, 'POST', '/session/export', { contextId: 'rt' })).json

    const b = buildTextMilkie(['ok'])
    const portB = await listen(createServeServer(b))
    const res = await request(portB, 'POST', '/session/import', { session: exported })
    expect(res.status).toBe(200)
    expect(res.json).toMatchObject({ contextId: 'rt' })
  })

  it('POST /session/import with an unsupported schemaVersion returns 400', async () => {
    const { milkie, agentId, broadcaster } = buildTextMilkie(['x'])
    const port = await listen(createServeServer({ milkie, agentId, broadcaster }))
    const bogus = { manifest: { schemaVersion: 999, contextId: 'x', agentId: 'echo', latestRunId: 'r', exportedAt: 0 }, events: [], variables: {} }
    const res = await request(port, 'POST', '/session/import', { session: bogus })
    expect(res.status).toBe(400)
  })
})
