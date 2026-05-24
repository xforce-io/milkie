import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { ReplayError } from '../trace/ReplayError'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

class SequentialGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const oneShotAgent = (agentId = 'a1'): AgentConfig => ({
  agentId,
  version: '0.0.0',
  systemPrompt: 'sys',
  fsm: {
    states: [
      { name: 'react', type: 'llm', instructions: 'say hi', tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
})

describe('Milkie.replay', () => {
  it('replays a recorded run with identical result and zero LLM calls', async () => {
    const store = new MemoryEventStore()
    const replayGateway = new SequentialGateway([text('this would be wrong')])
    // First run records
    const recordGateway = new SequentialGateway([text('hello world')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    // Replay reuses the same store + agent config but a different gateway
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store })
    replayMilkie.registerAgent(oneShotAgent())
    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)  // cache served everything
  })

  it('throws ReplayError when runId has no events', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new SequentialGateway([]),
      eventStore: new MemoryEventStore(),
    })
    await expect(milkie.replay('nonexistent')).rejects.toBeInstanceOf(ReplayError)
  })

  it('throws ReplayError when run has no lifecycle start (Phase 2 run)', async () => {
    const store = new MemoryEventStore()
    // Manually append only an llm.responded — no agent.run.started
    await store.append({
      id: 'e1', runId: 'r-old', type: 'llm.responded', actor: 'runtime', timestamp: 1,
      payload: { response: text('x'), requestHash: 'h' },
    })
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new SequentialGateway([]),
      eventStore: store,
    })
    await expect(milkie.replay('r-old')).rejects.toThrow(/no lifecycle start/)
  })

  it('throws ReplayError when agentId is not registered', async () => {
    const store = new MemoryEventStore()
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([text('x')]), eventStore: store })
    recordMilkie.registerAgent(oneShotAgent('a1'))
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    // intentionally do NOT register a1
    await expect(replayMilkie.replay(original.agentRunId)).rejects.toThrow(/not registered/)
  })

  it('throws ReplayDivergenceError when replay agent diverges from recorded I/O', async () => {
    const store = new MemoryEventStore()
    const recordGateway = new SequentialGateway([text('original')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    // Replay registers a *changed* agent so the LLM request differs
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    const mutated = oneShotAgent()
    ;(mutated.fsm.states[0] as { instructions?: string }).instructions = 'say goodbye'  // changes ModelRequest → hash mismatch
    replayMilkie.registerAgent(mutated)

    await expect(replayMilkie.replay(original.agentRunId)).rejects.toBeInstanceOf(ReplayDivergenceError)
  })

  it('does not write new events to the event store during replay (I7)', async () => {
    const store = new MemoryEventStore()
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([text('once')]), eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const before = (await store.readByRunId(original.agentRunId)).length
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    replayMilkie.registerAgent(oneShotAgent())
    await replayMilkie.replay(original.agentRunId)
    const after = (await store.readByRunId(original.agentRunId)).length

    expect(after).toBe(before)
  })
})
