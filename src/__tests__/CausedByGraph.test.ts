import { RecordingIOPort } from '../trace/RecordingIOPort'
import { CausalCursor } from '../trace/CausalCursor'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import type { IIOPort } from '../runtime/IOPort'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { Event, FsmTransitionPayload, RegionAddedPayload } from '../trace/types'
import type { AgentRunStartedPayload } from '../trace/types'
import type { AgentConfig } from '../types/agent'

class ExecInnerPort implements IIOPort {
  private nextClock = 1000
  private nextUuid  = 1
  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [], toolCalls: [], finishReason: 'end_turn' }
  }
  async invokeTool(_n: string, _i: unknown, execute: () => Promise<unknown>): Promise<unknown> {
    return execute()
  }
  now():  number { return this.nextClock++ }
  uuid(): string { return `uuid-${this.nextUuid++}` }
}

const START: AgentRunStartedPayload = {
  agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
}

const req = (): ModelRequest => ({ model: 'm', system: '', messages: [] })

async function events(store: MemoryEventStore): Promise<Event[]> {
  return store.readByRunId('r1')
}
const byType = (evs: Event[], t: string) => evs.filter(e => e.type === t)
const firstOf = (evs: Event[], t: string): Event => byType(evs, t)[0]!

describe('causedBy densify — RecordingIOPort edges (#30)', () => {
  it('edge 1: tool.requested.causedBy is the deciding llm.responded', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())

    await port.attach(START)
    await port.invokeLLM(req())
    await port.invokeTool('t', { a: 1 }, async () => 'out')

    const evs = await events(store)
    const llmResponded  = firstOf(evs, 'llm.responded')
    const toolRequested = firstOf(evs, 'tool.requested')
    expect(toolRequested.causedBy).toBe(llmResponded.id)
  })

  it('edge 2: first llm.requested.causedBy is agent.run.started; next is prior tool.responded', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())

    await port.attach(START)
    await port.invokeLLM(req())                                   // first llm.requested
    await port.invokeTool('t', {}, async () => 'out')             // tool.responded = terminator
    await port.invokeLLM(req())                                   // second llm.requested

    const evs = await events(store)
    const runStarted    = firstOf(evs, 'agent.run.started')
    const toolResponded = firstOf(evs, 'tool.responded')
    const llmRequested  = byType(evs, 'llm.requested')
    expect(llmRequested[0]!.causedBy).toBe(runStarted.id)
    expect(llmRequested[1]!.causedBy).toBe(toolResponded.id)
  })

  it('acceptance 1: any tool.requested walks causedBy back to agent.run.started', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())
    await port.attach(START)
    await port.invokeLLM(req())
    await port.invokeTool('t', {}, async () => 'out')

    const evs = await events(store)
    const byId = new Map(evs.map(e => [e.id, e]))
    let node: Event | undefined = firstOf(evs, 'tool.requested')
    const seen: string[] = []
    while (node) {
      seen.push(node.type)
      if (node.type === 'agent.run.started') break
      node = node.causedBy ? byId.get(node.causedBy) : undefined
    }
    expect(seen[seen.length - 1]).toBe('agent.run.started')
    expect(seen).toEqual(['tool.requested', 'llm.responded', 'llm.requested', 'agent.run.started'])
  })

  it('existing pairing edges unchanged (responded -> requested)', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())
    await port.attach(START)
    await port.invokeLLM(req())
    await port.invokeTool('t', {}, async () => 'out')

    const evs = await events(store)
    expect(firstOf(evs, 'llm.responded').causedBy).toBe(firstOf(evs, 'llm.requested').id)
    expect(firstOf(evs, 'tool.responded').causedBy).toBe(firstOf(evs, 'tool.requested').id)
  })

  it('completion edge: agent.run.completed.causedBy is the final llm.responded', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())

    await port.attach(START)
    await port.invokeLLM(req())
    await port.detach({ status: 'completed' })

    const evs = await events(store)
    const llmResponded = firstOf(evs, 'llm.responded')
    const completed    = firstOf(evs, 'agent.run.completed')
    expect(completed.causedBy).toBe(llmResponded.id)
  })

  it('agent.run.completed has no causedBy when the run made no LLM call', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())

    await port.attach(START)
    await port.detach({ status: 'completed' })

    const completed = firstOf(await events(store), 'agent.run.completed')
    expect(completed.causedBy).toBeUndefined()
  })

  it('cursor omitted: no new causedBy edges, no throw', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')   // no cursor
    await port.attach(START)
    await port.invokeLLM(req())
    await port.invokeTool('t', {}, async () => 'out')

    const evs = await events(store)
    expect(firstOf(evs, 'llm.requested').causedBy).toBeUndefined()
    expect(firstOf(evs, 'tool.requested').causedBy).toBeUndefined()
  })
})

class StubGateway implements IModelGateway {
  constructor(private responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('No more stub responses')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}
const textResponse = (text: string): ModelResponse => ({
  content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn',
})
const twoStateConfig = (): AgentConfig => ({
  agentId:      'multi-state',
  version:      '1.0.0',
  systemPrompt: 'sys',
  fsm: { states: [
    { name: 'react', type: 'llm', on: { DONE: 'wrap' } },
    { name: 'wrap',  type: 'action', terminal: true },
  ] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
})

describe('causedBy densify — AgentRuntime edges (#30)', () => {
  it('edge 3 + edge 4: fsm.transition <- llm.responded; crystallized region.added <- boundary', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('answer')]),
      eventStore,
    })
    milkie.registerAgent(twoStateConfig())
    const result = await milkie.invoke({ agentId: 'multi-state', goal: 'g', input: 'i' })
    expect(result.status).toBe('completed')

    const evs = await eventStore.readByRunId(result.agentRunId)

    // edge 3: the react->wrap DONE transition was triggered by the text llm.responded
    const transition = evs.find(e => e.type === 'fsm.transition') as Event<FsmTransitionPayload>
    const llmResponded = firstOf(evs, 'llm.responded')
    expect(transition.causedBy).toBe(llmResponded.id)

    // edge 4: the crystallized history-pair region was caused by the turn-end boundary
    const boundary = firstOf(evs, 'context.boundary.applied')
    const historyAdded = (evs.filter(e => e.type === 'region.added') as Event<RegionAddedPayload>[])
      .find(e => e.payload.section === 'history')
    expect(historyAdded).toBeDefined()
    expect(historyAdded!.causedBy).toBe(boundary.id)

    // agent-set exception: the initial header region has no upstream boundary
    const headerAdded = (evs.filter(e => e.type === 'region.added') as Event<RegionAddedPayload>[])
      .find(e => e.payload.id === 'header')
    expect(headerAdded!.causedBy).toBeUndefined()
  })

  it('sub-agent run has its own cursor: child llm.requested traces to child agent.run.started', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new StubGateway([
        { content: [{ type: 'tool_use', id: 's1', name: 'worker', input: { goal: 'sg', input: 'si' } }],
          toolCalls: [{ id: 's1', name: 'worker', input: { goal: 'sg', input: 'si' } }], finishReason: 'tool_use' },
        textResponse('worker done'),
        textResponse('all done'),
      ]),
      eventStore,
    })
    milkie.registerAgent({
      agentId: 'supervisor', version: '0.0.0', systemPrompt: 's',
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' }, subAgents: { worker: '1.0.0' },
    })
    milkie.registerAgent({
      agentId: 'worker', version: '0.0.0', systemPrompt: 's',
      fsm: { states: [{ name: 'react', type: 'llm' }] },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' },
    })

    const result = await milkie.invoke({ agentId: 'supervisor', goal: 'g', input: 'i' })
    const parentEvs = await eventStore.readByRunId(result.agentRunId)
    const childRunId = (parentEvs.find(e => e.type === 'agent.spawned')!.payload as { childRunId: string }).childRunId
    const childEvs   = await eventStore.readByRunId(childRunId)

    // child's first llm.requested traces back to the CHILD's own run root, not the parent's
    const childRunStarted = firstOf(childEvs, 'agent.run.started')
    const childLlmReq      = firstOf(childEvs, 'llm.requested')
    expect(childLlmReq.causedBy).toBe(childRunStarted.id)
    // isolation: parent ids never appear as a child causedBy
    const parentIds = new Set(parentEvs.map(e => e.id))
    for (const e of childEvs) {
      if (e.causedBy) expect(parentIds.has(e.causedBy)).toBe(false)
    }
  })
})
