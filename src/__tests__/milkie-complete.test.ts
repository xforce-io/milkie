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
