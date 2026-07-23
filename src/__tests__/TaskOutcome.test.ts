/**
 * Unit tests for #217 / s-016 task outcome record + query.
 * Drives real Milkie.recordTaskOutcome / getTaskOutcome entry points.
 */

import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import {
  TaskOutcomeError,
  TaskOutcomeRunNotFoundError,
} from '../types/outcome'
import type { AgentRunCompletedPayload, TaskOutcomeRecordedPayload } from '../trace/types'

class FixedGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly text: string) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    return {
      content:      [{ type: 'text', text: this.text }],
      toolCalls:    [],
      finishReason: 'end_turn',
    }
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

const agent: AgentConfig = {
  agentId:      'outcome-unit-agent',
  version:      '0.0.0',
  systemPrompt: 'answer briefly',
  fsm: {
    states: [
      { name: 'reply', type: 'llm', terminal: true, tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

describe('TaskOutcome (#217 / s-016)', () => {
  let eventStore: MemoryEventStore
  let milkie: Milkie
  let gateway: FixedGateway

  beforeEach(() => {
    eventStore = new MemoryEventStore()
    gateway    = new FixedGateway('hello')
    milkie     = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway,
    })
    milkie.registerAgent(agent)
  })

  it('records failure post-hoc and reads it back without changing execution status', async () => {
    const result = await milkie.invoke({
      agentId: 'outcome-unit-agent',
      goal:    'say hi',
      input:   'hi',
    })
    expect(result.status).toBe('completed')
    expect(result.agentRunId).toBeTruthy()
    const runId = result.agentRunId
    const callsAfterInvoke = gateway.callCount

    const recorded = await milkie.recordTaskOutcome({
      runId,
      value:  'failure',
      source: 'eval',
      note:   'wrong answer for eval case',
      scores: [{ name: 'accuracy', value: 0 }],
    })
    expect(recorded.value).toBe('failure')
    expect(recorded.source).toBe('eval')
    expect(recorded.scores).toEqual([{ name: 'accuracy', value: 0 }])

    const got = await milkie.getTaskOutcome(runId)
    expect(got).not.toBeNull()
    expect(got!.value).toBe('failure')
    expect(got!.source).toBe('eval')
    expect(got!.scores).toEqual([{ name: 'accuracy', value: 0 }])
    expect(got!.note).toBe('wrong answer for eval case')

    // execution status unchanged
    expect(result.status).toBe('completed')
    const events = await eventStore.readByRunId(runId)
    const completed = events.find(e => e.type === 'agent.run.completed')
    expect(completed).toBeDefined()
    expect((completed!.payload as AgentRunCompletedPayload).status).toBe('completed')
    const outcomeEvents = events.filter(e => e.type === 'task.outcome.recorded')
    expect(outcomeEvents).toHaveLength(1)
    expect((outcomeEvents[0]!.payload as TaskOutcomeRecordedPayload).value).toBe('failure')

    // no extra LLM
    expect(gateway.callCount).toBe(callsAfterInvoke)

    // eslint-disable-next-line no-console
    console.log(
      `outcome-smoke: invoke.status=${result.status} outcome=${got!.value} completedEvent=${(completed!.payload as AgentRunCompletedPayload).status}`,
    )
  })

  it('returns null for a known run with no outcome; throws for unknown runId', async () => {
    const result = await milkie.invoke({
      agentId: 'outcome-unit-agent',
      goal:    'say hi',
      input:   'hi',
    })
    await expect(milkie.getTaskOutcome(result.agentRunId)).resolves.toBeNull()

    await expect(milkie.getTaskOutcome('run-does-not-exist')).rejects.toBeInstanceOf(
      TaskOutcomeRunNotFoundError,
    )
  })

  it('rejects empty source; last write wins', async () => {
    const result = await milkie.invoke({
      agentId: 'outcome-unit-agent',
      goal:    'say hi',
      input:   'hi',
    })
    const runId = result.agentRunId

    await expect(
      milkie.recordTaskOutcome({ runId, value: 'failure', source: '  ' }),
    ).rejects.toBeInstanceOf(TaskOutcomeError)

    await milkie.recordTaskOutcome({ runId, value: 'failure', source: 'human' })
    await milkie.recordTaskOutcome({ runId, value: 'success', source: 'eval' })
    const got = await milkie.getTaskOutcome(runId)
    expect(got!.value).toBe('success')
    expect(got!.source).toBe('eval')
  })

  it('requires eventStore', async () => {
    const bare = new Milkie({ stateStore: new MemoryStore(), gateway })
    bare.registerAgent(agent)
    const result = await bare.invoke({
      agentId: 'outcome-unit-agent',
      goal:    'x',
      input:   'x',
    })
    await expect(
      bare.recordTaskOutcome({ runId: result.agentRunId, value: 'failure', source: 'eval' }),
    ).rejects.toBeInstanceOf(TaskOutcomeError)
  })
})
