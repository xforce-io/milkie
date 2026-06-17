import { promises as fs } from 'fs'
import os from 'os'
import path from 'path'

import { MemoryEventStore } from '../trace/MemoryEventStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import { FileTraceObjectStore, MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import { contextBefore, getRegionAt } from '../trace/RegionContextView'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { canonicalize, contentAddressForCanonicalBytes } from '../trace/hash'
import type {
  Event,
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
  RegionAddedPayload,
  SkillLifecyclePayload,
  AgentSpawnedPayload,
  AgentReturnedPayload,
  AgentRunStartedPayload,
  AgentRunCompletedPayload,
} from '../trace/types'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

class FailingEventStore extends MemoryEventStore {
  async append(event: Event): Promise<void> {
    if (event.type === 'region.added' || event.type === 'region.removed' || event.type === 'context.boundary.applied') {
      throw new Error('disk full')
    }
    await super.append(event)
  }
}

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

const toolCallResponse = (id: string, name: string, input: unknown): ModelResponse => ({
  content:      [{ type: 'tool_use', id, name, input }],
  toolCalls:    [{ id, name, input }],
  finishReason: 'tool_use',
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

// ---- TraceObjectStore / region context reconstruction ----

describe('TraceObjectStore', () => {
  it('deduplicates identical canonical bytes and reads them by content hash', async () => {
    const store = new MemoryTraceObjectStore()

    const h1 = await store.putCanonical('{"a":1}')
    const h2 = await store.putCanonical('{"a":1}')

    expect(h1).toBe(h2)
    expect(await store.getCanonical(h1)).toBe('{"a":1}')
  })

  it('FileTraceObjectStore round-trips canonical bytes across instances', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-objects-'))
    try {
      const writer = new FileTraceObjectStore(dir)
      const hash = await writer.putCanonical('{"name":"research"}')

      const reader = new FileTraceObjectStore(dir)
      expect(await reader.getCanonical(hash)).toBe('{"name":"research"}')
    } finally {
      await fs.rm(dir, { recursive: true, force: true })
    }
  })

  it('reconstructs context regions from region lifecycle events plus object store', async () => {
    const objects = new MemoryTraceObjectStore()
    const h1 = await objects.putCanonical('"v1"')
    const h2 = await objects.putCanonical('"v2"')

    const events: Event[] = [
      makeEvent({
        id: 'e1',
        type: 'region.added',
        payload: { id: 'skill:research', target: 'system', section: 'session-skills', stability: 'turn-stable', reason: 'test', contentHash: h1 },
      }),
      makeEvent({
        id: 'e2',
        type: 'region.added',
        payload: { id: 'skill:research', target: 'system', section: 'session-skills', stability: 'turn-stable', reason: 'test', contentHash: h2 },
      }),
      makeEvent({
        id: 'e3',
        type: 'region.removed',
        payload: { id: 'skill:research', reason: 'test' },
      }),
    ]

    expect((await getRegionAt(events, 'e1', 'skill:research', objects))?.content).toBe('"v1"')
    expect((await getRegionAt(events, 'e2', 'skill:research', objects))?.content).toBe('"v2"')
    expect(await getRegionAt(events, 'e3', 'skill:research', objects)).toBeUndefined()
  })

  it('gracefully hydrates legacy region.added events without contentHash', async () => {
    const objects = new MemoryTraceObjectStore()
    const events: Event[] = [
      makeEvent({
        id: 'legacy',
        type: 'region.added',
        payload: { id: 'header', target: 'system', section: 'header', stability: 'immutable', reason: 'legacy' },
      }),
    ]

    expect((await getRegionAt(events, 'legacy', 'header', objects))?.content).toBeUndefined()
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

    await port.attach({
      agentId:   'a1',
      goal:      'do the thing',
      input:     'go',
      contextId: 'ctx-1',
    })

    const events = await store.readByRunId('r1')
    const started = events.find(e => e.type === 'agent.run.started')!
    expect(started.payload).toEqual({
      agentId: 'a1', goal: 'do the thing', input: 'go', contextId: 'ctx-1', parentId: undefined,
    })
  })

  it('detach emits agent.run.completed with terminal status', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    await port.attach({ agentId: 'a1', goal: 'g', input: 'i', contextId: 'c1' })
    await port.detach({ status: 'completed', lastTextOutput: 'done' })

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
    // clock.read / uuid.generated nondet events may appear before agent.run.started
    // (ContextRegions.set() calls clock in the AgentRuntime constructor before
    // attach() is called). Use find() instead of positional assertions.
    const startedEvt = events.find(e => e.type === 'agent.run.started')
    expect(startedEvt).toBeDefined()
    expect(kinds[kinds.length - 1]).toBe('agent.run.completed')
    expect((startedEvt!.payload as { agentId: string }).agentId).toBe('a1')
  })

  it('region.added events carry content hashes and can reconstruct context before an LLM request', async () => {
    const eventStore = new MemoryEventStore()
    const traceObjectStore = new MemoryTraceObjectStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hello')]),
      eventStore,
      traceObjectStore,
    })
    milkie.registerAgent(minimalAgentConfig('a1'))

    const result = await milkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const events = await eventStore.readByRunId(result.agentRunId)
    const headerAdded = events.find(e => e.type === 'region.added' && (e.payload as RegionAddedPayload).id === 'header')!
    const headerPayload = headerAdded.payload as RegionAddedPayload
    expect(headerPayload.contentHash).toMatch(/^sha256:[a-f0-9]{64}$/)
    expect(await traceObjectStore.getCanonical(headerPayload.contentHash!)).toBe('"system"')

    const firstLlm = events.find(e => e.type === 'llm.requested')!
    const context = await contextBefore(events, firstLlm.id, traceObjectStore)
    expect(context.get('header')?.content).toBe('"system"')
    expect(context.get('current-turn')?.content).toContain('Goal: g')
  })

  it('does not expose content hashes when no traceObjectStore is configured', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hello')]),
      eventStore,
    })
    milkie.registerAgent(minimalAgentConfig('a1'))

    const result = await milkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })
    const events = await eventStore.readByRunId(result.agentRunId)
    const added = events.find(e => e.type === 'region.added')!

    expect((added.payload as RegionAddedPayload).contentHash).toBeUndefined()
  })

  it('trace write failures do not change the agent result', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hello')]),
      eventStore: new FailingEventStore(),
      traceObjectStore: new MemoryTraceObjectStore(),
    })
    milkie.registerAgent(minimalAgentConfig('a1'))

    const result = await milkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    expect(result.status).toBe('completed')
    expect(result.output).toBe('hello')
  })

  it('records skill.loaded and skill.unloaded lifecycle events with version provenance', async () => {
    const store = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new StubGateway([
        toolCallResponse('skill-1', 'skill_request', { name: 'research' }),
        textResponse('done with research'),
      ]),
      eventStore: store,
    })
    milkie.registerAgent({
      ...minimalAgentConfig('test-agent'),
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
      skills: { research: '1.0.0' },
      skillInstructions: { research: 'Research skill instructions' },
    })

    const result = await milkie.invoke({ agentId: 'test-agent', goal: 'g', input: 'i' })

    const events = await store.readByRunId(result.agentRunId)
    const loaded = events.find(e => e.type === 'skill.loaded')!
    const unloaded = events.find(e => e.type === 'skill.unloaded')!
    expect(loaded).toBeDefined()
    expect(unloaded).toBeDefined()
    const loadedIndex = events.findIndex(e => e.type === 'skill.loaded')
    const requestedWithSkillIndex = events.findIndex(e =>
      e.type === 'llm.requested'
      && ((e.payload as LlmRequestedPayload).request.system ?? '').includes('Research skill instructions')
    )
    expect(requestedWithSkillIndex).toBeGreaterThan(-1)
    expect(loadedIndex).toBeLessThan(requestedWithSkillIndex)

    const loadedPayload = loaded.payload as SkillLifecyclePayload
    expect(loadedPayload).toMatchObject({
      skillId: 'skill:research',
      version: '1.0.0',
      source:  'agent-config.skillInstructions',
    })
    expect(loadedPayload.sha).toMatch(/^[a-f0-9]{64}$/)
    expect(unloaded.payload).toEqual(loaded.payload)
  })
})

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

    const spawned  = events[spawnedIdx]!.payload as AgentSpawnedPayload
    const returned = events[returnedIdx]!.payload as AgentReturnedPayload
    expect(spawned.parentRunId).toBe(result.agentRunId)
    expect(spawned.agentId).toBe('worker')
    expect(spawned.goal).toBe('subgoal')
    expect(returned.status).toBe('completed')
    expect(returned.childRunId).toBe(spawned.childRunId)
  })

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

    const spawned  = events.find(e => e.type === 'agent.spawned')!.payload as AgentSpawnedPayload
    const returned = events.find(e => e.type === 'agent.returned')!.payload as AgentReturnedPayload
    expect(returned.status).toBe('error')
    expect(returned.childRunId).toBe(spawned.childRunId)
  })

  it('records child LLM I/O under an independent childRunId, not the parent run', async () => {
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
    expect(spawned.childRunId).not.toBe(result.agentRunId)

    const parentLlm = parentEvents.filter(e => e.type === 'llm.requested')
    const childEvents = await eventStore.readByRunId(spawned.childRunId)
    const childLlm = childEvents.filter(e => e.type === 'llm.requested')
    expect(childLlm.length).toBeGreaterThan(0)
    expect(parentLlm.length).toBe(2)
  })

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
    expect(childEvents[0]!.type).toBe('agent.run.started')
    expect((started.payload as AgentRunStartedPayload).parentId).toBe(result.agentRunId)
    expect((started.payload as AgentRunStartedPayload).agentId).toBe('worker')
    expect((completed.payload as AgentRunCompletedPayload).status).toBe('completed')
    expect(started.runId).toBe(spawned.childRunId)
  })

  it('child port factory resolves gateway from child config', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    const make = (milkie as any)['buildMakeChildPort']()
    expect(make).not.toBeNull()
    expect(make).not.toBeUndefined()
    const bogus = { ...workerConfig(), model: { provider: 'x', model: 'm', adapter: 'no-such-adapter' } }
    await expect(make('child-run-1', bogus, {
      agentId: 'worker', goal: 'g', input: 'i', contextId: 'c', parentId: 'p',
    })).rejects.toThrow(/no-such-adapter/)
  })

  it('child sub-agent tool output is written through to the trace object store', async () => {
    const eventStore  = new MemoryEventStore()
    const objectStore = new MemoryTraceObjectStore()
    const big = { chapter: 'x'.repeat(50_000) }
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new StubGateway([
        toolCallResponse('s1', 'worker',   { goal: 'g', input: 'i' }),  // supervisor spawns worker
        toolCallResponse('w1', 'echo_big', {}),                         // worker calls echo_big
        textResponse('worker done'),                                    // worker finishes
        textResponse('all done'),                                       // supervisor finishes
      ]),
      eventStore,
      traceObjectStore: objectStore,
      tools: [{
        name:         'echo_big',
        description:  'returns a big object',
        parallelSafe: true,
        inputSchema:  { type: 'object', properties: {} },
        handler:      async () => big,
      }],
    })
    milkie.registerAgent(supervisorConfig())
    milkie.registerAgent(workerConfig())

    await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })

    // echo_big is only ever called by the worker (child) — its output reaching the
    // object store proves the child port was wired with traceObjectStore.
    const hash = contentAddressForCanonicalBytes(canonicalize(big))
    expect(await objectStore.has(hash)).toBe(true)
  })
})

// ---- decision anchor (#175 de-core) ----
//
// The old `fsm.transition events` describe block (#21) asserted the runtime wrote
// one `fsm.transition` business-topology event per authored polymorphic transition.
// #175 de-cores the multi-state FSM: the runtime no longer writes `fsm.transition`,
// and the decision anchor moved onto the `llm.responded` / `tool.responded` effects
// (explainDecision). Those pure transition-write assertions are deleted. The
// surviving "this run still completes" coverage is folded into the assertion below,
// which also confirms the decision-effect anchor is present in the log.
describe('decision anchor lives on effects, not fsm.transition', () => {
  const twoStateConfig = (): AgentConfig => ({
    agentId:      'multi-state',
    version:      '1.0.0',
    systemPrompt: 'sys',
    fsm: {
      states: [
        { name: 'react', type: 'llm',    on: { DONE: 'wrap' } },
        { name: 'wrap',  type: 'action', terminal: true },
      ],
    },
    model: { provider: 'test', model: 'test', adapter: 'test' },
  })

  it('records no fsm.transition event; the llm.responded effect is the decision anchor', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hi')]),
      eventStore,
    })
    milkie.registerAgent(twoStateConfig())

    const result = await milkie.invoke({ agentId: 'multi-state', goal: 'g', input: 'i' })
    expect(result.status).toBe('completed')

    const events = await eventStore.readByRunId(result.agentRunId)
    expect(events.filter(e => e.type === 'fsm.transition')).toHaveLength(0)
    // the decision anchor (#175) is the llm.responded effect, present in the log
    expect(events.some(e => e.type === 'llm.responded')).toBe(true)
  })
})
