import { promises as fs } from 'fs'
import os from 'os'
import path from 'path'

import { MemoryEventStore } from '../trace/MemoryEventStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import type {
  Event,
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
} from '../trace/types'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

function minimalAgentConfig(agentId: string): AgentConfig {
  return {
    agentId,
    version:      '0.0.0',
    systemPrompt: 'system',
    fsm: {
      states: [
        { name: 'react', type: 'llm', instructions: 'reply hello', tools: [] },
      ],
    },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  }
}

// ---- Helpers ----

function makeEvent(over: Partial<Event> = {}): Event {
  return {
    id:        'e1',
    runId:     'r1',
    type:      'llm.requested',
    actor:     'runtime',
    timestamp: 1,
    payload:   { request: {} },
    ...over,
  }
}

class StubGateway implements IModelGateway {
  private responses: ModelResponse[]
  constructor(responses: ModelResponse[]) { this.responses = responses }
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('No more stub responses')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const textResponse = (text: string): ModelResponse => ({
  content:      [{ type: 'text', text }],
  toolCalls:    [],
  finishReason: 'end_turn',
})

const sampleRequest: ModelRequest = {
  model:    'test',
  system:   'test system',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
}

// ---- MemoryEventStore ----

describe('MemoryEventStore', () => {
  it('appends and reads back in order', async () => {
    const store = new MemoryEventStore()
    await store.append(makeEvent({ id: 'e1' }))
    await store.append(makeEvent({ id: 'e2' }))
    const events = await store.readByRunId('r1')
    expect(events.map(e => e.id)).toEqual(['e1', 'e2'])
  })

  it('readByRunId returns [] for unknown run', async () => {
    const store = new MemoryEventStore()
    expect(await store.readByRunId('nope')).toEqual([])
  })

  it('returned arrays are copies (mutation does not corrupt store)', async () => {
    const store = new MemoryEventStore()
    await store.append(makeEvent({ id: 'e1' }))
    const events = await store.readByRunId('r1')
    events.push(makeEvent({ id: 'tampered' }))
    const fresh = await store.readByRunId('r1')
    expect(fresh.map(e => e.id)).toEqual(['e1'])
  })

  it('readRange respects fromIndex and count', async () => {
    const store = new MemoryEventStore()
    for (let i = 0; i < 5; i++) await store.append(makeEvent({ id: `e${i}` }))
    expect((await store.readRange('r1', 1, 2)).map(e => e.id)).toEqual(['e1', 'e2'])
    expect((await store.readRange('r1', 3)).map(e => e.id)).toEqual(['e3', 'e4'])
  })
})

// ---- JsonlEventStore ----

describe('JsonlEventStore', () => {
  let dir: string

  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-jsonl-'))
  })

  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true })
  })

  it('appends to per-run file and reads back', async () => {
    const store = new JsonlEventStore(dir)
    await store.append(makeEvent({ id: 'e1', runId: 'run-a' }))
    await store.append(makeEvent({ id: 'e2', runId: 'run-a' }))
    await store.append(makeEvent({ id: 'e3', runId: 'run-b' }))

    expect((await store.readByRunId('run-a')).map(e => e.id)).toEqual(['e1', 'e2'])
    expect((await store.readByRunId('run-b')).map(e => e.id)).toEqual(['e3'])
  })

  it('readByRunId returns [] when file does not exist', async () => {
    const store = new JsonlEventStore(dir)
    expect(await store.readByRunId('nonexistent')).toEqual([])
  })

  it('creates baseDir lazily', async () => {
    const nested = path.join(dir, 'a', 'b', 'c')
    const store = new JsonlEventStore(nested)
    await store.append(makeEvent())
    const stat = await fs.stat(nested)
    expect(stat.isDirectory()).toBe(true)
  })
})

// ---- RecordingIOPort ----

describe('RecordingIOPort', () => {
  it('emits paired llm.requested / llm.responded events with causedBy chain', async () => {
    const store = new MemoryEventStore()
    const inner = new DefaultIOPort(new StubGateway([textResponse('reply')]))
    const recording = new RecordingIOPort(inner, store, 'run-1')

    const resp = await recording.invokeLLM(sampleRequest)
    expect(resp.content[0]).toMatchObject({ type: 'text', text: 'reply' })

    const events = await store.readByRunId('run-1')
    expect(events.map(e => e.type)).toEqual(['llm.requested', 'llm.responded'])

    const [reqEvent, respEvent] = events
    expect(reqEvent?.runId).toBe('run-1')
    expect(respEvent?.causedBy).toBe(reqEvent?.id)
    expect((reqEvent?.payload as LlmRequestedPayload).request.model).toBe('test')
    expect((respEvent?.payload as LlmRespondedPayload).response.content[0])
      .toMatchObject({ type: 'text', text: 'reply' })
  })

  it('emits paired tool events for successful invocation', async () => {
    const store = new MemoryEventStore()
    const inner = new DefaultIOPort(new StubGateway([]))
    const recording = new RecordingIOPort(inner, store, 'run-2')

    const result = await recording.invokeTool(
      'echo',
      { msg: 'hi' },
      async () => ({ echoed: 'hi' }),
    )
    expect(result).toEqual({ echoed: 'hi' })

    const events = await store.readByRunId('run-2')
    expect(events.map(e => e.type)).toEqual(['tool.requested', 'tool.responded'])

    const [reqEvent, respEvent] = events
    expect((reqEvent?.payload as ToolRequestedPayload).toolName).toBe('echo')
    expect((reqEvent?.payload as ToolRequestedPayload).input).toEqual({ msg: 'hi' })
    expect((respEvent?.payload as ToolRespondedPayload).output).toEqual({ echoed: 'hi' })
    expect((respEvent?.payload as ToolRespondedPayload).error).toBeUndefined()
    expect(respEvent?.causedBy).toBe(reqEvent?.id)
  })

  it('emits tool.responded with error when invocation throws', async () => {
    const store = new MemoryEventStore()
    const inner = new DefaultIOPort(new StubGateway([]))
    const recording = new RecordingIOPort(inner, store, 'run-3')

    await expect(
      recording.invokeTool('broken', {}, async () => { throw new Error('boom') }),
    ).rejects.toThrow('boom')

    const events = await store.readByRunId('run-3')
    expect(events.map(e => e.type)).toEqual(['tool.requested', 'tool.responded'])
    expect((events[1]?.payload as ToolRespondedPayload).error?.message).toBe('boom')
    expect((events[1]?.payload as ToolRespondedPayload).output).toBeUndefined()
  })

  it('now() and uuid() pass through to inner port', () => {
    const store = new MemoryEventStore()
    const inner = new DefaultIOPort(new StubGateway([]))
    const recording = new RecordingIOPort(inner, store, 'run-4')

    expect(typeof recording.now()).toBe('number')
    expect(recording.uuid()).toMatch(/^[0-9a-f-]+$/)
  })
})

// ---- Milkie auto-wrap ----

describe('Milkie eventStore integration', () => {
  const baseConfig = (overrides: Partial<AgentConfig> = {}): AgentConfig => ({
    agentId:      'test-agent',
    version:      '1.0.0',
    systemPrompt: 'You are a test agent.',
    fsm:          { states: [{ name: 'react', type: 'llm' }] },
    model:        { provider: 'test', model: 'test', adapter: 'test' },
    ...overrides,
  })

  it('auto-wraps RecordingIOPort when eventStore is provided', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hi')]),
      eventStore,
    })
    milkie.registerAgent(baseConfig())

    const result = await milkie.invoke({
      agentId: 'test-agent',
      goal:    'g',
      input:   'i',
    })

    expect(result.status).toBe('completed')
    const events = await eventStore.readByRunId(result.agentRunId)
    // At least one LLM request/response pair
    expect(events.map(e => e.type)).toContain('llm.requested')
    expect(events.map(e => e.type)).toContain('llm.responded')
  })

  it('does not record when eventStore is omitted', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hi')]),
      // no eventStore
    })
    milkie.registerAgent(baseConfig())

    const result = await milkie.invoke({
      agentId: 'test-agent',
      goal:    'g',
      input:   'i',
    })
    expect(result.status).toBe('completed')
    // No store to inspect — just verify the run still works without recording
  })
})

// ---- Phase 3 additions ----

describe('RecordingIOPort — Phase 3 additions', () => {
  it('LLM events carry requestHash matching hashModelRequest(request)', async () => {
    const { hashModelRequest } = await import('../trace/hash')
    const store = new MemoryEventStore()
    const gateway = new StubGateway([textResponse('hi')])
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'r1')

    const req: ModelRequest = {
      model:    'm1',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      system:   'sys',
      tools:    [],
    }
    await port.invokeLLM(req)

    const events = await store.readByRunId('r1')
    const requested = events.find(e => e.type === 'llm.requested')!
    expect((requested.payload as { requestHash: string }).requestHash)
      .toBe(hashModelRequest(req))
  })

  it('tool error is recorded as structured payload preserving retryable/code/name', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    const err = Object.assign(new Error('boom'), { retryable: true, code: 'EBUSY', name: 'BusyError' })
    await expect(port.invokeTool('t', { x: 1 }, async () => { throw err })).rejects.toThrow('boom')

    const events = await store.readByRunId('r1')
    const responded = events.find(e => e.type === 'tool.responded')!
    const payload = responded.payload as { error?: { message: string; retryable?: boolean; code?: string; name?: string } }
    expect(payload.error).toEqual({ message: 'boom', retryable: true, code: 'EBUSY', name: 'BusyError' })
  })

  it('attach emits agent.run.started with lifecycle identity payload', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    port.attach({
      agentId:   'a1',
      goal:      'do the thing',
      input:     'go',
      contextId: 'ctx-1',
      parentId:  undefined,
    })

    // attach uses store.append (async); flush microtasks
    await new Promise(resolve => setImmediate(resolve))

    const events = await store.readByRunId('r1')
    const started = events.find(e => e.type === 'agent.run.started')!
    expect(started.payload).toEqual({
      agentId: 'a1', goal: 'do the thing', input: 'go', contextId: 'ctx-1', parentId: undefined,
    })
  })

  it('detach emits agent.run.completed with terminal status', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    port.attach({ agentId: 'a1', goal: 'g', input: 'i', contextId: 'c1' })
    port.detach({ status: 'completed', lastTextOutput: 'done' })
    await new Promise(resolve => setImmediate(resolve))

    const events = await store.readByRunId('r1')
    const completed = events.find(e => e.type === 'agent.run.completed')!
    expect(completed.payload).toEqual({ status: 'completed', lastTextOutput: 'done' })
  })

  it('Milkie.invoke wraps run with attach/detach around the run', async () => {
    const store = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hello')]),
      eventStore: store,
    })
    milkie.registerAgent(minimalAgentConfig('a1'))

    const result = await milkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const events = await store.readByRunId(result.agentRunId)
    const kinds = events.map(e => e.type)
    expect(kinds[0]).toBe('agent.run.started')
    expect(kinds[kinds.length - 1]).toBe('agent.run.completed')
    expect((events[0]!.payload as { agentId: string }).agentId).toBe('a1')
  })
})
