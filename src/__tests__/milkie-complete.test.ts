// #124: Milkie.complete — a one-shot LLM completion that bypasses the FSM.
// It resolves an agent's gateway/model and calls it directly (no agent run, no
// event log), mirroring DefaultIOPort.invokeLLM: non-streaming when onEvent is
// omitted, streaming + aggregation when provided.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type { Message } from '../types/common'

function textGateway(deltas: string[]): IModelGateway {
  const full = deltas.join('')
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text: full }], toolCalls: [], usage: { inputTokens: 3, outputTokens: 5 }, finishReason: 'end_turn' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      for (const d of deltas) yield { type: 'message_delta', data: { text: d } }
      yield { type: 'usage', data: { inputTokens: 3, outputTokens: 5 } }
    },
  }
}

const agent: AgentConfig = {
  agentId: 'echo', version: '1.0.0', systemPrompt: 'echo',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model: { provider: 'stub', model: 'stub-model', adapter: 'stub' },
}

function buildMilkie(gw: IModelGateway): Milkie {
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: gw })
  milkie.registerAgent(agent)
  return milkie
}

const msgs: Message[] = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]

// #126: a gateway that records every ModelRequest it receives, so tests can
// assert which model/tier was resolved and whether temperature was forwarded.
function capturingGateway(): { gw: IModelGateway; reqs: ModelRequest[] } {
  const reqs: ModelRequest[] = []
  const gw: IModelGateway = {
    async complete(req: ModelRequest): Promise<ModelResponse> {
      reqs.push(req)
      return { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(req: ModelRequest): AsyncIterable<ModelEvent> {
      reqs.push(req)
      yield { type: 'message_delta', data: { text: 'ok' } }
    },
  }
  return { gw, reqs }
}

// Agent with a default tier plus a named `fast` tier (alfred's compressor档).
const tieredAgent: AgentConfig = {
  agentId: 'tiered', version: '1.0.0', systemPrompt: 'sys',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model:  { provider: 'stub', model: 'default-model', adapter: 'stub' },
  models: { fast: { provider: 'stub', model: 'fast-model', adapter: 'stub' } },
}

function buildTieredMilkie(gw: IModelGateway): Milkie {
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: gw })
  milkie.registerAgent(tieredAgent)
  return milkie
}

describe('#126 Milkie.complete — tier + temperature', () => {
  it('tier="fast" resolves the fast tier model into the ModelRequest', async () => {
    const { gw, reqs } = capturingGateway()
    await buildTieredMilkie(gw).complete('tiered', { messages: msgs, tier: 'fast' })
    expect(reqs[0]!.model).toBe('fast-model')
  })

  it('omitted tier resolves the default model', async () => {
    const { gw, reqs } = capturingGateway()
    await buildTieredMilkie(gw).complete('tiered', { messages: msgs })
    expect(reqs[0]!.model).toBe('default-model')
  })

  it('unknown tier falls back to the default model (does not throw)', async () => {
    const { gw, reqs } = capturingGateway()
    await buildTieredMilkie(gw).complete('tiered', { messages: msgs, tier: 'nonexistent' })
    expect(reqs[0]!.model).toBe('default-model')
  })

  it('temperature flows into the ModelRequest; omitted leaves it undefined', async () => {
    const { gw, reqs } = capturingGateway()
    const milkie = buildTieredMilkie(gw)
    await milkie.complete('tiered', { messages: msgs, temperature: 0.2 })
    await milkie.complete('tiered', { messages: msgs })
    expect(reqs[0]!.temperature).toBe(0.2)
    expect(reqs[1]!.temperature).toBeUndefined()
  })
})

describe('#124 Milkie.complete', () => {
  it('non-streaming: returns the aggregated response', async () => {
    const milkie = buildMilkie(textGateway(['Hello, ', 'world!']))
    const res = await milkie.complete('echo', { messages: msgs })
    expect(res.content).toEqual([{ type: 'text', text: 'Hello, world!' }])
    expect(res.usage).toMatchObject({ outputTokens: 5 })
  })

  it('streaming: forwards token deltas to onEvent and aggregates the same result', async () => {
    const milkie = buildMilkie(textGateway(['Hel', 'lo']))
    const deltas: string[] = []
    const res = await milkie.complete('echo', { messages: msgs }, e => {
      if (e.type === 'message_delta') deltas.push(e.data.text)
    })
    expect(deltas).toEqual(['Hel', 'lo'])
    expect(res.content).toEqual([{ type: 'text', text: 'Hello' }])
  })

  it('throws when the agent is unknown', async () => {
    const milkie = buildMilkie(textGateway(['x']))
    await expect(milkie.complete('nope', { messages: msgs })).rejects.toThrow(/nope/)
  })
})
