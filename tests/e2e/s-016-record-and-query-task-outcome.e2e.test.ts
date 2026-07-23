/**
 * s-016: Record and query task outcome after a run.
 *
 * Hermetic: stub gateway. Asserts:
 *   - invoke → completed + runId
 *   - post-hoc recordTaskOutcome(failure) + optional score
 *   - getTaskOutcome reads back the same value/source
 *   - agent.run.completed.status stays completed (invariant 14)
 *   - known run without outcome → null
 *   - unknown runId → TaskOutcomeRunNotFoundError
 *   - no extra LLM on record/query
 */

import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'
import { TaskOutcomeRunNotFoundError } from '../../src/types/outcome'
import type { AgentRunCompletedPayload } from '../../src/trace/types'

class SequentialGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

const textOnly = (s: string): ModelResponse => ({
  content:      [{ type: 'text', text: s }],
  toolCalls:    [],
  finishReason: 'end_turn',
})

const agent: AgentConfig = {
  agentId:      's016-outcome-agent',
  version:      '0.0.0',
  systemPrompt: 'you recommend a supplier',
  fsm: {
    states: [
      { name: 'answer', type: 'llm', terminal: true, tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

describe('s-016 record and query task outcome', () => {
  let eventStore: MemoryEventStore
  let milkie: Milkie
  let gateway: SequentialGateway

  beforeAll(async () => {
    eventStore = new MemoryEventStore()
    gateway    = new SequentialGateway([textOnly('Recommend supplier A')])
    milkie     = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway,
    })
    milkie.registerAgent(agent)
  })

  it('records failure after completed run; status stays completed; query matches', async () => {
    const result = await milkie.invoke({
      agentId: 's016-outcome-agent',
      goal:    'recommend a supplier',
      input:   'who should we buy from?',
    })

    expect(result.status).toBe('completed')
    expect(result.agentRunId).toBeTruthy()
    const runId = result.agentRunId
    const llmCalls = gateway.callCount

    const recorded = await milkie.recordTaskOutcome({
      runId,
      value:  'failure',
      source: 'eval',
      note:   'supplier A is wrong per gold set',
      scores: [{ name: 'gold_match', value: false }],
    })
    expect(recorded.value).toBe('failure')
    expect(recorded.source).toBe('eval')

    const outcome = await milkie.getTaskOutcome(runId)
    expect(outcome).not.toBeNull()
    expect(outcome!.value).toBe('failure')
    expect(outcome!.source).toBe('eval')
    expect(outcome!.scores).toEqual([{ name: 'gold_match', value: false }])

    // completed 与 failure 共存
    expect(result.status).toBe('completed')
    const events = await eventStore.readByRunId(runId)
    const completedEv = events.find(e => e.type === 'agent.run.completed')
    expect(completedEv).toBeDefined()
    expect((completedEv!.payload as AgentRunCompletedPayload).status).toBe('completed')

    expect(gateway.callCount).toBe(llmCalls)

    // eslint-disable-next-line no-console
    console.log(
      `s-016 smoke: status=${result.status} outcome=${outcome!.value} runId=${runId}`,
    )
  })

  it('null for known run without outcome; error for unknown runId', async () => {
    // second run with no outcome recorded
    gateway = new SequentialGateway([textOnly('ok')])
    milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway,
    })
    milkie.registerAgent(agent)

    const bare = await milkie.invoke({
      agentId: 's016-outcome-agent',
      goal:    'noop',
      input:   'hi',
    })
    await expect(milkie.getTaskOutcome(bare.agentRunId)).resolves.toBeNull()
    await expect(milkie.getTaskOutcome('never-seen-run-id')).rejects.toBeInstanceOf(
      TaskOutcomeRunNotFoundError,
    )
  })
})
