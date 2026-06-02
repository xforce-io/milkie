// #84: portable session export/import — the event log is the single source of
// truth (#73), so a portable session bundles the run-tree's events + the
// context's persistent vars (#83) + a versioned manifest. Importing into a fresh
// Milkie instance and invoking the same contextId must let the agent continue
// with prior history.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type { ToolDefinition } from '../types/tool'
import type { AgentSpawnedPayload, AgentRunStartedPayload } from '../trace/types'

/** A gateway that answers immediately (end_turn) — a turn that completes still
 *  emits a continuation checkpoint event (see checkpoint-from-events.test). */
function endTurnGateway(text = 'done'): IModelGateway {
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
  }
}

/** Writes a fact into working memory; used to prove history carry-forward. */
function factWriter(): ToolDefinition {
  return {
    name: 'record_fact', description: 'record a fact',
    inputSchema: { type: 'object', properties: { key: { type: 'string' }, value: { type: 'string' } }, required: ['key', 'value'] },
    handler: async (input: unknown, ctx) => {
      const { key, value } = input as { key: string; value: string }
      ctx.workingMemory.set(key, value)
      return { recorded: key }
    },
  }
}

/** First call records a fact (tool_use), then completes (end_turn). */
function recordThenDoneGateway(): IModelGateway {
  let n = 0
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      n++
      if (n === 1) {
        const input = { key: 'fact1', value: 'v1' }
        return { content: [{ type: 'tool_use', id: 'c1', name: 'record_fact', input }], toolCalls: [{ id: 'c1', name: 'record_fact', input }], finishReason: 'tool_use' }
      }
      return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
  }
}

const recorderAgent: AgentConfig = {
  agentId: 'recorder', version: '1.0.0', systemPrompt: 'answer',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 50 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

describe('#84 portable session export', () => {
  it('exports a versioned payload with the run-tree events and context vars', async () => {
    const stateStore = new MemoryStore()
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({ stateStore, eventStore, gateway: endTurnGateway() })
    milkie.registerAgent(recorderAgent)

    const contextId = 'ctx-export'
    const run = await milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'hello', contextId })
    expect(run.status).toBe('completed')
    await milkie.setContextVar(contextId, 'user', 'alice')

    const session = await milkie.exportSession(contextId)

    expect(session.manifest.schemaVersion).toBe(1)
    expect(session.manifest.contextId).toBe(contextId)
    expect(session.manifest.agentId).toBe('recorder')
    expect(session.manifest.latestRunId).toBe(run.agentRunId)
    expect(typeof session.manifest.exportedAt).toBe('number')

    // events bundle the latest run's stream, including its continuation checkpoint.
    expect(session.events.some(e => e.runId === run.agentRunId)).toBe(true)
    expect(checkpointFromEvents(session.events.filter(e => e.runId === run.agentRunId))).not.toBeNull()

    // vars are captured as a snapshot.
    expect(session.variables).toEqual({ user: 'alice' })
  })

  it('throws when exporting an unknown contextId', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: endTurnGateway() })
    milkie.registerAgent(recorderAgent)
    await expect(milkie.exportSession('nope')).rejects.toThrow()
  })
})

describe('#84 portable session round-trip (single agent)', () => {
  it('imports into a fresh Milkie so the agent continues with prior history + vars', async () => {
    // Source instance: one turn that records a fact into WM, then completes.
    const srcState = new MemoryStore()
    const srcEvents = new MemoryEventStore()
    const src = new Milkie({ stateStore: srcState, eventStore: srcEvents, gateway: recordThenDoneGateway(), tools: [factWriter()] })
    src.registerAgent(recorderAgent)

    const contextId = 'ctx-rt'
    const run1 = await src.invoke({ agentId: 'recorder', goal: 'g', input: 'remember v1', contextId })
    expect(run1.status).toBe('completed')
    await src.setContextVar(contextId, 'user', 'alice')

    const session = await src.exportSession(contextId)

    // Destination instance: brand-new stores, nothing carried over implicitly.
    const dstState = new MemoryStore()
    const dstEvents = new MemoryEventStore()
    const dst = new Milkie({ stateStore: dstState, eventStore: dstEvents, gateway: endTurnGateway('continued'), tools: [factWriter()] })
    dst.registerAgent(recorderAgent)

    const { contextId: restoredId } = await dst.importSession(session)
    expect(restoredId).toBe(contextId)

    // Vars are restored into the destination stateStore.
    expect(await dst.getContextVar(contextId, 'user')).toBe('alice')

    // Continue the conversation: the next turn's first prompt must carry the
    // prior turn's WM (restored from the imported event log).
    const run2 = await dst.invoke({ agentId: 'recorder', goal: 'g', input: 'continue', contextId })
    const ev2 = await dstEvents.readByRunId(run2.agentRunId)
    const firstLlm = ev2.find(e => e.type === 'llm.requested')!
    const prompt = JSON.stringify((firstLlm.payload as { request: unknown }).request)
    expect(prompt).toContain('fact1')
  })
})

// ---- multi-agent: a supervisor that spawns a worker sub-agent ----

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
const textResponse = (text: string): ModelResponse => ({ content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn' })
const toolCallResponse = (id: string, name: string, input: unknown): ModelResponse => ({ content: [{ type: 'tool_use', id, name, input }], toolCalls: [{ id, name, input }], finishReason: 'tool_use' })

const supervisorAgent: AgentConfig = {
  agentId: 'supervisor', version: '1.0.0', systemPrompt: 'system',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  subAgents: { worker: '1.0.0' },
}
const workerAgent: AgentConfig = {
  agentId: 'worker', version: '1.0.0', systemPrompt: 'system',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
}

/** A supervisor turn that spawns one worker, run on the given Milkie. */
function spawningGateway(): IModelGateway {
  return new StubGateway([
    toolCallResponse('s1', 'worker', { goal: 'subgoal', input: 'subinput' }),
    textResponse('worker done'),   // worker's only LLM turn
    textResponse('all done'),      // supervisor finishes
  ])
}

describe('#84 portable session (multi-agent)', () => {
  it('export captures the sub-agent run-tree (child events under their own runId)', async () => {
    const stateStore = new MemoryStore()
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({ stateStore, eventStore, gateway: spawningGateway() })
    milkie.registerAgent(supervisorAgent)
    milkie.registerAgent(workerAgent)

    const contextId = 'ctx-ma'
    const run = await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i', contextId })
    expect(run.status).toBe('completed')

    // discover the child runId from the parent's spawn event
    const parentEvents = await eventStore.readByRunId(run.agentRunId)
    const childRunId = (parentEvents.find(e => e.type === 'agent.spawned')!.payload as AgentSpawnedPayload).childRunId

    const session = await milkie.exportSession(contextId)

    // both parent and child run events are bundled
    expect(session.events.some(e => e.runId === run.agentRunId)).toBe(true)
    expect(session.events.some(e => e.runId === childRunId)).toBe(true)
    // the child's own lifecycle is present, pointing back at the parent
    const childStarted = session.events.find(e => e.runId === childRunId && e.type === 'agent.run.started')!
    expect((childStarted.payload as AgentRunStartedPayload).parentId).toBe(run.agentRunId)
    expect((childStarted.payload as AgentRunStartedPayload).agentId).toBe('worker')
  })

  it('round-trips the sub-agent tree: child run is replayable in a fresh instance', async () => {
    // source
    const srcEvents = new MemoryEventStore()
    const src = new Milkie({ stateStore: new MemoryStore(), eventStore: srcEvents, gateway: spawningGateway() })
    src.registerAgent(supervisorAgent)
    src.registerAgent(workerAgent)
    const contextId = 'ctx-ma-rt'
    const run = await src.invoke({ agentId: 'supervisor', goal: 'g', input: 'i', contextId })
    const srcParentEvents = await srcEvents.readByRunId(run.agentRunId)
    const childRunId = (srcParentEvents.find(e => e.type === 'agent.spawned')!.payload as AgentSpawnedPayload).childRunId

    const session = await src.exportSession(contextId)

    // destination: fresh stores, no live gateway responses needed for replay
    const dstEvents = new MemoryEventStore()
    const dst = new Milkie({ stateStore: new MemoryStore(), eventStore: dstEvents, gateway: new StubGateway([]) })
    dst.registerAgent(supervisorAgent)
    dst.registerAgent(workerAgent)

    await dst.importSession(session)

    // the child run's full event stream landed in the destination store
    const dstChildEvents = await dstEvents.readByRunId(childRunId)
    expect(dstChildEvents.length).toBeGreaterThan(0)
    expect(dstChildEvents.some(e => e.type === 'agent.run.started')).toBe(true)

    // and it replays byte-identically from the imported events (no live LLM)
    const replayed = await dst.replay(childRunId)
    expect(replayed.status).toBe('completed')
  })
})

describe('#84 portable session (multi-turn + versioning)', () => {
  /** Records fact{n} on the first call of each turn, completes on the second. */
  function multiTurnRecorder(): IModelGateway {
    let calls = 0, facts = 0
    return {
      async complete(_req: ModelRequest): Promise<ModelResponse> {
        calls++
        if (calls % 2 === 1) {
          facts++
          const input = { key: `fact${facts}`, value: `v${facts}` }
          return { content: [{ type: 'tool_use', id: `c${facts}`, name: 'record_fact', input }], toolCalls: [{ id: `c${facts}`, name: 'record_fact', input }], finishReason: 'tool_use' }
        }
        return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
      },
      async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
    }
  }

  it('carries multi-turn history forward through export/import', async () => {
    const src = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: multiTurnRecorder(), tools: [factWriter()] })
    src.registerAgent(recorderAgent)
    const contextId = 'ctx-mt'
    await src.invoke({ agentId: 'recorder', goal: 'g', input: 'turn1', contextId })  // → fact1
    await src.invoke({ agentId: 'recorder', goal: 'g', input: 'turn2', contextId })  // → fact2

    const session = await src.exportSession(contextId)

    const dstEvents = new MemoryEventStore()
    const dst = new Milkie({ stateStore: new MemoryStore(), eventStore: dstEvents, gateway: endTurnGateway('next'), tools: [factWriter()] })
    dst.registerAgent(recorderAgent)
    await dst.importSession(session)

    const run3 = await dst.invoke({ agentId: 'recorder', goal: 'g', input: 'turn3', contextId })
    const ev3 = await dstEvents.readByRunId(run3.agentRunId)
    const firstLlm = ev3.find(e => e.type === 'llm.requested')!
    const prompt = JSON.stringify((firstLlm.payload as { request: unknown }).request)
    // both turns' facts survive the round-trip (accumulated working memory)
    expect(prompt).toContain('fact1')
    expect(prompt).toContain('fact2')
  })

  it('rejects an unknown schemaVersion on import', async () => {
    const dst = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: endTurnGateway() })
    dst.registerAgent(recorderAgent)
    const bad = {
      manifest: { schemaVersion: 999, contextId: 'x', agentId: 'recorder', latestRunId: 'r', exportedAt: 0 },
      events: [],
      variables: {},
    } as unknown as Parameters<typeof dst.importSession>[0]
    await expect(dst.importSession(bad)).rejects.toThrow(/schemaVersion/)
  })

  it('rejects import when this instance has no eventStore', async () => {
    const dst = new Milkie({ stateStore: new MemoryStore(), gateway: endTurnGateway(), defaultModel: { provider: 'test', model: 'test', adapter: 'test' } })
    const empty = { manifest: { schemaVersion: 1, contextId: 'x', agentId: 'recorder', latestRunId: 'r', exportedAt: 0 }, events: [], variables: {} } as Parameters<typeof dst.importSession>[0]
    await expect(dst.importSession(empty)).rejects.toThrow(/eventStore/)
  })
})
