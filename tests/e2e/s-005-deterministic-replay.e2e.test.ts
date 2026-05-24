import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'

/**
 * s-005: Deterministically replay a recorded agent run.
 *
 * Phase 3 scope: structural replay. Result must equal original on
 * status + output; LLM gateway must not be called during replay.
 */

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

const twoStepAgent: AgentConfig = {
  agentId: 'replay-demo',
  version: '0.0.0',
  systemPrompt: 'you are a friendly bot',
  fsm: {
    states: [
      { name: 'greet',    type: 'llm', instructions: 'say hello',           tools: [], on: { DONE: 'farewell' } },
      { name: 'farewell', type: 'llm', instructions: 'say goodbye briefly', tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

describe('s-005 deterministic replay (Phase 3 structural)', () => {
  test('record a 2-step run, then replay it without a live gateway', async () => {
    const eventStore = new MemoryEventStore()

    // ---- Record ----
    const recordGateway = new SequentialGateway([text('Hello!'), text('Goodbye!')])
    const recordMilkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    recordGateway,
      eventStore,
    })
    recordMilkie.registerAgent(twoStepAgent)
    const original = await recordMilkie.invoke({
      agentId: 'replay-demo', goal: 'demo replay', input: 'start',
    })

    expect(original.status).toBe('completed')
    expect(recordGateway.callCount).toBe(2)

    // ---- Replay ----
    const replayGateway = new SequentialGateway([])  // empty: must not be called
    const replayMilkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    replayGateway,
      eventStore,
    })
    replayMilkie.registerAgent(twoStepAgent)

    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)
  })
})
