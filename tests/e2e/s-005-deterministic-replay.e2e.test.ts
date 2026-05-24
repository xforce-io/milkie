import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'

/**
 * s-005: Deterministically replay a recorded agent run.
 *
 * Phase 4 scope: byte-identical replay. Result must equal original on
 * status + output; live LLM gateway must not be called during replay;
 * recorded clock.read / uuid.generated events are served from the cache
 * so replay never re-samples nondet. The Phase-4 cache+nondet contract
 * is enforced strictly by Milkie.replay's tail check — any unconsumed
 * event throws ReplayDivergenceError.
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

describe('s-005 deterministic replay (Phase 4 byte-identical)', () => {
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

    // Phase 4: recording captured nondet events. The actual count depends on
    // the runtime's port.now/port.uuid calls along the recorded path; we just
    // assert at least one was captured, which is enough to know the recording
    // path went through RecordingIOPort's pending-buffer + flush mechanism.
    const events = await eventStore.readByRunId(original.agentRunId)
    const nondets = events.filter(e => e.type === 'clock.read' || e.type === 'uuid.generated')
    expect(nondets.length).toBeGreaterThan(0)

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

  test('replay is repeatable: re-running against the same recording produces identical results', async () => {
    const eventStore = new MemoryEventStore()
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

    // Replay 3 times back-to-back. Without Phase 4 nondet recording, each
    // replay would re-sample timestamps/UUIDs internally; if any of those
    // values affected agent-observable state, results would drift. Phase 4
    // guarantees byte-identical: all three runs produce the same status +
    // output, and none of them invoke the live gateway.
    const replays: Array<{ status: string; output: string }> = []
    for (let i = 0; i < 3; i++) {
      const replayMilkie = new Milkie({
        stateStore: new MemoryStore(),
        gateway:    new SequentialGateway([]),
        eventStore,
      })
      replayMilkie.registerAgent(twoStepAgent)
      const r = await replayMilkie.replay(original.agentRunId)
      replays.push({ status: r.status, output: r.output })
    }
    expect(new Set(replays.map(r => JSON.stringify(r))).size).toBe(1)
    expect(replays[0]!.status).toBe(original.status)
    expect(replays[0]!.output).toBe(original.output)
  })
})
